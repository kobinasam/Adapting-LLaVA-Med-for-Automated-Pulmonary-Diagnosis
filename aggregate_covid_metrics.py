"""
Compute BLEU, ROUGE, BERTScore for one or more predictions.json files produced
by evaluate_covid.py, then aggregate across seeds for mean +/- std.

Usage:
    # Single model
    python aggregate_covid_metrics.py \\
        --predictions ./eval_covid_seed42/predictions.json \\
        --label "COVID-FT (s=42)"

    # Multiple seeds -> aggregated mean+/-std
    python aggregate_covid_metrics.py \\
        --predictions ./eval_covid_seed42/predictions.json ./eval_covid_seed1234/predictions.json ./eval_covid_seed5678/predictions.json \\
        --labels "COVID-FT (s=42)" "COVID-FT (s=1234)" "COVID-FT (s=5678)" \\
        --aggregate_label "COVID-FT (mean+/-std, n=3)"

Outputs a table of per-seed scores and (if multiple seeds) the aggregated row.
Also writes a LaTeX-ready table to <output_dir>/table.tex if --output_dir is given.
"""
import argparse
import json
import os
import statistics

import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def load_predictions(path):
    with open(path) as f:
        rows = json.load(f)
    preds, refs = [], []
    for r in rows:
        if r.get("prediction") and r.get("ground_truth"):
            preds.append(r["prediction"].strip())
            refs.append(r["ground_truth"].strip())
    return preds, refs


def compute_bleu(preds, refs):
    sf = SmoothingFunction().method1
    refs_tokenized = [[r.split()] for r in refs]
    preds_tokenized = [p.split() for p in preds]
    return {
        "BLEU-1": corpus_bleu(refs_tokenized, preds_tokenized,
                              weights=(1, 0, 0, 0), smoothing_function=sf),
        "BLEU-2": corpus_bleu(refs_tokenized, preds_tokenized,
                              weights=(0.5, 0.5, 0, 0), smoothing_function=sf),
        "BLEU-3": corpus_bleu(refs_tokenized, preds_tokenized,
                              weights=(1/3, 1/3, 1/3, 0), smoothing_function=sf),
        "BLEU-4": corpus_bleu(refs_tokenized, preds_tokenized,
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf),
    }


def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return {
        "ROUGE-1": float(np.mean(r1)),
        "ROUGE-2": float(np.mean(r2)),
        "ROUGE-L": float(np.mean(rL)),
    }


def compute_bertscore(preds, refs):
    P, R, F1 = bert_score(preds, refs, lang="en", verbose=False, batch_size=32)
    return {
        "BERTScore-P": float(P.mean().item()),
        "BERTScore-R": float(R.mean().item()),
        "BERTScore-F1": float(F1.mean().item()),
    }


def compute_all(preds, refs):
    print(f"  Scoring {len(preds)} samples...")
    out = {}
    out.update(compute_bleu(preds, refs))
    out.update(compute_rouge(preds, refs))
    out.update(compute_bertscore(preds, refs))
    return out


def print_table(rows, keys_to_report):
    """Pretty-print a table of per-model scores."""
    name_w = max(len(r["label"]) for r in rows)
    print(f"\n{'Model':<{name_w}}  " + "  ".join(f"{k:>12}" for k in keys_to_report))
    print("-" * (name_w + 2 + 14 * len(keys_to_report)))
    for r in rows:
        row = f"{r['label']:<{name_w}}  " + "  ".join(
            f"{r['metrics'][k]:>12.4f}" for k in keys_to_report
        )
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Human-readable label per predictions file")
    ap.add_argument("--aggregate_label", default=None,
                    help="Label for the mean+/-std row when 2+ predictions files are given")
    ap.add_argument("--output_dir", default=None,
                    help="Write results.json and table.tex here")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.predictions):
        raise SystemExit("--labels must match length of --predictions")
    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.predictions]

    key_order = [
        "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
        "ROUGE-1", "ROUGE-2", "ROUGE-L",
        "BERTScore-P", "BERTScore-R", "BERTScore-F1",
    ]

    rows = []
    for path, label in zip(args.predictions, labels):
        print(f"\n== {label} ({path}) ==")
        preds, refs = load_predictions(path)
        metrics = compute_all(preds, refs)
        rows.append({"label": label, "path": path, "n": len(preds), "metrics": metrics})

    # Aggregate across seeds
    aggregate_row = None
    if len(rows) >= 2:
        agg_label = args.aggregate_label or f"mean+/-std (n={len(rows)})"
        agg_metrics = {}
        for k in key_order:
            vals = [r["metrics"][k] for r in rows]
            agg_metrics[k + "_mean"] = float(np.mean(vals))
            agg_metrics[k + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        aggregate_row = {"label": agg_label, "n_seeds": len(rows), "metrics": agg_metrics}

    # Print
    print_table(rows, key_order)

    if aggregate_row:
        print(f"\n== Aggregated across {len(rows)} seeds ==")
        name_w = len(aggregate_row["label"])
        print(f"{'Metric':<15}  {'Mean':>10}  {'Std':>10}")
        print("-" * 40)
        for k in key_order:
            m = aggregate_row["metrics"][k + "_mean"]
            s = aggregate_row["metrics"][k + "_std"]
            print(f"{k:<15}  {m:>10.4f}  {s:>10.4f}")

    # Save
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump({"per_seed": rows, "aggregated": aggregate_row}, f, indent=2)

        # LaTeX ready row for Table I (BLEU-4, ROUGE-L, BERTScore F1)
        if aggregate_row:
            m = aggregate_row["metrics"]
            latex_row = (
                f"COVID-FT (mean$\\pm$std, n={aggregate_row['n_seeds']}) & "
                f"{m['BLEU-1_mean']:.3f}$\\pm${m['BLEU-1_std']:.3f} & "
                f"{m['BLEU-4_mean']:.3f}$\\pm${m['BLEU-4_std']:.3f} & "
                f"{m['ROUGE-L_mean']:.3f}$\\pm${m['ROUGE-L_std']:.3f} & "
                f"{m['BERTScore-F1_mean']:.3f}$\\pm${m['BERTScore-F1_std']:.3f} \\\\\n"
            )
            with open(os.path.join(args.output_dir, "table_row.tex"), "w") as f:
                f.write(latex_row)
            print(f"\nLaTeX row written to {os.path.join(args.output_dir, 'table_row.tex')}:")
            print(latex_row)


if __name__ == "__main__":
    main()
