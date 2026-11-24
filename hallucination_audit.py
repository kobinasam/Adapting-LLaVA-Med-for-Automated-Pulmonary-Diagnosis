"""
Quantifies the hallucination pattern on the UCSD external test set:
how often does the model describe a non-COVID scan as COVID-consistent,
and how often does it miss a true COVID scan.

For each prediction, parses a binary COVID label from the text:
  positive := says the scan is consistent with COVID-19 pneumonia
  negative := says the scan is not consistent with COVID-19 pneumonia
  unparseable := neither clear stance

Reports per-seed and aggregated:
  - accuracy, sensitivity, specificity
  - false-positive rate (the main hallucination number)
  - unparseable rate
And dumps the 10 worst false positives verbatim to fp_examples.txt
so the paper can include qualitative evidence.

Usage:
    python hallucination_audit.py \\
        --predictions \\
            /scratch/bhwr/msam1/LLaVA-Med/eval_ucsd_seed42/predictions.json \\
            /scratch/bhwr/msam1/LLaVA-Med/eval_ucsd_seed1234/predictions.json \\
            /scratch/bhwr/msam1/LLaVA-Med/eval_ucsd_seed5678/predictions.json \\
        --labels s42 s1234 s5678 \\
        --output_dir /scratch/bhwr/msam1/LLaVA-Med/eval_ucsd_hallucination
"""
import argparse
import json
import os
import re
import statistics


NEG_PATTERNS = [
    r"not\s+consistent\s+with\s+covid",
    r"no\s+evidence\s+of\s+covid",
    r"does\s+not\s+(show|suggest|indicate)\s+covid",
    r"inconsistent\s+with\s+covid",
    r"unlikely\s+to\s+be\s+covid",
    r"no\s+(signs?|features?)\s+of\s+covid",
]

POS_PATTERNS = [
    r"consistent\s+with\s+covid",
    r"suggestive\s+of\s+covid",
    r"indicative\s+of\s+covid",
    r"compatible\s+with\s+covid",
    r"characteristic\s+of\s+covid",
    r"typical\s+of\s+covid",
    r"covid[-\s]19\s+pneumonia",
]


def parse_label(text):
    """Return 'POS', 'NEG', or 'UNK' from free-form text."""
    t = text.lower()
    for pat in NEG_PATTERNS:
        if re.search(pat, t):
            return "NEG"
    for pat in POS_PATTERNS:
        if re.search(pat, t):
            return "POS"
    return "UNK"


def truth_from_image(image_name):
    if image_name.startswith("covid_"):
        return "POS"
    if image_name.startswith("noncovid_"):
        return "NEG"
    return "UNK"


def score_one(rows):
    tp = fp = tn = fn = 0
    unparseable = 0
    fp_rows = []
    fn_rows = []
    for r in rows:
        y = truth_from_image(r["image"])
        yhat = parse_label(r["prediction"])
        if yhat == "UNK":
            unparseable += 1
            continue
        if y == "POS" and yhat == "POS":
            tp += 1
        elif y == "NEG" and yhat == "NEG":
            tn += 1
        elif y == "NEG" and yhat == "POS":
            fp += 1
            fp_rows.append(r)
        elif y == "POS" and yhat == "NEG":
            fn += 1
            fn_rows.append(r)

    n_scored = tp + tn + fp + fn
    n_total = n_scored + unparseable
    acc = (tp + tn) / n_scored if n_scored else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (tn + fp) if (tn + fp) else 0.0

    return {
        "n_total": n_total,
        "n_scored": n_scored,
        "unparseable": unparseable,
        "unparseable_rate": unparseable / n_total if n_total else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "fpr_hallucination_rate": fpr,
        "fp_examples": fp_rows,
        "fn_examples": fn_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.predictions]
    assert len(labels) == len(args.predictions)

    per_seed = []
    for path, label in zip(args.predictions, labels):
        with open(path) as f:
            rows = json.load(f)
        result = score_one(rows)
        result["label"] = label
        result["path"] = path
        per_seed.append(result)

    numeric_keys = ["accuracy", "sensitivity", "specificity",
                    "fpr_hallucination_rate", "unparseable_rate"]

    print(f"\n{'Seed':<10}  {'N':>5}  {'Acc':>6}  {'Sens':>6}  {'Spec':>6}  "
          f"{'FPR':>6}  {'Unpars':>6}")
    print("-" * 60)
    for r in per_seed:
        print(f"{r['label']:<10}  {r['n_scored']:>5}  "
              f"{r['accuracy']:>6.3f}  {r['sensitivity']:>6.3f}  "
              f"{r['specificity']:>6.3f}  {r['fpr_hallucination_rate']:>6.3f}  "
              f"{r['unparseable_rate']:>6.3f}")

    agg = {}
    for k in numeric_keys:
        vals = [r[k] for r in per_seed]
        agg[k + "_mean"] = statistics.mean(vals)
        agg[k + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0

    print("-" * 60)
    print(f"{'MEAN':<10}         {agg['accuracy_mean']:>6.3f}  "
          f"{agg['sensitivity_mean']:>6.3f}  {agg['specificity_mean']:>6.3f}  "
          f"{agg['fpr_hallucination_rate_mean']:>6.3f}  "
          f"{agg['unparseable_rate_mean']:>6.3f}")
    print(f"{'STD':<10}         {agg['accuracy_std']:>6.3f}  "
          f"{agg['sensitivity_std']:>6.3f}  {agg['specificity_std']:>6.3f}  "
          f"{agg['fpr_hallucination_rate_std']:>6.3f}  "
          f"{agg['unparseable_rate_std']:>6.3f}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump({
                "per_seed": [
                    {k: v for k, v in r.items() if k not in ("fp_examples", "fn_examples")}
                    for r in per_seed
                ],
                "aggregated": agg,
            }, f, indent=2)

        with open(os.path.join(args.output_dir, "fp_examples.txt"), "w") as f:
            for r in per_seed:
                f.write(f"\n==== {r['label']} false positives (hallucinated COVID on non-COVID) ====\n")
                for ex in r["fp_examples"][:10]:
                    f.write(f"\n[{ex['image']}]\n")
                    f.write(f"PRED: {ex['prediction'][:500]}\n")

        latex = (
            f"COVID-FT UCSD (mean$\\pm$std, n={len(per_seed)}) & "
            f"{agg['accuracy_mean']:.3f}$\\pm${agg['accuracy_std']:.3f} & "
            f"{agg['sensitivity_mean']:.3f}$\\pm${agg['sensitivity_std']:.3f} & "
            f"{agg['specificity_mean']:.3f}$\\pm${agg['specificity_std']:.3f} & "
            f"{agg['fpr_hallucination_rate_mean']:.3f}$\\pm${agg['fpr_hallucination_rate_std']:.3f} \\\\\n"
        )
        with open(os.path.join(args.output_dir, "table_row.tex"), "w") as f:
            f.write(latex)
        print(f"\nWrote {args.output_dir}/results.json, fp_examples.txt, table_row.tex")


if __name__ == "__main__":
    main()
