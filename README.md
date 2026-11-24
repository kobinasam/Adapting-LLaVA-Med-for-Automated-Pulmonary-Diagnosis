# Adapting LLaVA-Med for automated pulmonary diagnosis

This repository holds code and results for a COVID-19 chest CT vision-language model built on LLaVA-Med. Given a CT slice and a natural-language question, the model produces a free-text clinical description.

The main finding concerns evaluation more than the model itself. On an external cohort the model collapses to a single class, and the usual text-similarity metrics and LLM judges miss it. The hallucination-audit script in this repo is what exposes the collapse, and you can run it on your own cohort.

## Method

Training uses a two-stage curriculum:

1. Alignment. A trainable projection module maps features from a frozen CLIP ViT-L/14 vision encoder into the embedding space of Vicuna-7B.
2. Instruction tuning. The projection module and the language model are trained jointly on about 14,000 curated COVID-19 CT image-question-answer triples.

Stage 2 was run with three random seeds (42, 1234, 5678). Every number below is mean ± std over those seeds.

## Results

### Text similarity

| Test set | N | BLEU-4 | ROUGE-L | BERTScore F1 |
|---|---|---|---|---|
| CIDC held-out (intra-cohort) | 1,429 | 0.176 ± 0.002 | 0.362 ± 0.002 | 0.912 ± 0.000 |
| UCSD COVID-CT (external, patient-disjoint) | 200 | not reported | not reported | 0.887 ± 0.001 |

External BERTScore F1 drops by only 2.7%, which on its face suggests the model transfers reasonably well.

### Hallucination audit (UCSD external set)

Parsing each prediction for a binary COVID stance gives a different result:

| Metric | Value |
|---|---|
| Sensitivity | 1.000 ± 0.000 |
| Specificity | 0.000 ± 0.000 |
| Unparseable (no stance) | 37.7 ± 2.4% |

Every parseable prediction on a non-COVID scan uses COVID-consistent language. The model has collapsed to one class on the unseen cohort, and aggregate BERTScore gives no sign of it.

### Judge agreement

An LLM-as-Judge evaluation with `llama3.1:8b` gives a mean score of 8.09/10 across accuracy, completeness, and clinical relevance. When a radiologist re-scored 25 samples blind, inter-rater agreement with the LLM judge was near zero or negative on all three criteria.

## Repository layout

| Path | Contents |
|---|---|
| `llava/` | LLaVA / LLaVA-Med model, training (`llava/train/`), evaluation, and serving code |
| `evaluate_covid.py` | Batch inference on a test JSON, writes `predictions.json` (resumable, greedy decoding) |
| `aggregate_covid_metrics.py` | BLEU, ROUGE, and BERTScore per seed plus mean ± std; optional LaTeX table |
| `build_ucsd_test.py` | Builds the balanced 200-image UCSD COVID-CT test set (100 COVID, 100 non-COVID) |
| `hallucination_audit.py` | Binary stance parsing, sensitivity/specificity/FPR/unparseable rate, worst false positives |
| `demo_gradio_covid.py` | Gradio demo for the fine-tuned COVID model, with PDF report export |
| `demo_gradio_compare.py` | Same demo pointed at the base LLaVA-Med checkpoint, for side-by-side comparison |
| `eval_results_instruct/` | Evaluation predictions and scores |
| `benchmark_results/` | Benchmark comparison charts and LaTeX tables |
| `captum_results/` | Captum attribution maps (integrated gradients, feature ablation, layer attribution) |

Model checkpoints and the CIDC training data are not included.

## Usage

Paths below are placeholders. Replace them with your own.

### 1. Run inference

```bash
python evaluate_covid.py \
    --val_json     data/llava_test.json \
    --image_folder data/images_covid \
    --model_path   checkpoints/stage2-covid \
    --output_dir   ./eval_covid_seed42
```

Optional flags: `--max_tokens` (default 400) and `--limit` to run on a subset.

### 2. Score with text metrics

```bash
python aggregate_covid_metrics.py \
    --predictions ./eval_covid_seed42/predictions.json \
                  ./eval_covid_seed1234/predictions.json \
                  ./eval_covid_seed5678/predictions.json \
    --labels "s42" "s1234" "s5678" \
    --aggregate_label "COVID-FT (mean ± std, n=3)" \
    --output_dir ./metrics
```

### 3. Build the external UCSD test set

Download the [UCSD COVID-CT dataset](https://github.com/UCSD-AI4H/COVID-CT) (He et al., 2020), then:

```bash
python build_ucsd_test.py \
    --covid_dir    COVID-CT/Images-processed/CT_COVID \
    --noncovid_dir COVID-CT/Images-processed/CT_NonCOVID \
    --output       ucsd_test.json \
    --n_per_class  100 \
    --seed         42
```

UCSD provides only binary labels, so ground truth is a per-class template sentence. Text metrics on this set measure agreement with that template, not with image-specific reports.

### 4. Run the hallucination audit

```bash
python hallucination_audit.py \
    --predictions eval_ucsd_seed42/predictions.json \
                  eval_ucsd_seed1234/predictions.json \
                  eval_ucsd_seed5678/predictions.json \
    --labels s42 s1234 s5678 \
    --output_dir ./hallucination_audit
```

The script reports accuracy, sensitivity, specificity, false-positive rate, and unparseable rate for each seed and in aggregate. It also writes the 10 worst false positives to `fp_examples.txt`.

### 5. Launch the demo

Set `BASE_MODEL_PATH` near the top of the script to your checkpoint, then:

```bash
python demo_gradio_covid.py
```

The demo serves on port 7860.

## Acknowledgments

Built on [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) and [LLaVA](https://github.com/haotian-liu/LLaVA). External evaluation uses the [UCSD COVID-CT](https://github.com/UCSD-AI4H/COVID-CT) dataset.

## Citation

Citation details will be added once the paper is published.
