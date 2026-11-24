"""
Build a LLaVA-schema test JSON from the UCSD COVID-CT dataset
(He et al. 2020, https://github.com/UCSD-AI4H/COVID-CT).

Samples a balanced 200-image subset (100 COVID + 100 non-COVID) matching
your paper's N=200 test size, writes JSON in the schema evaluate_covid.py reads.

Ground truth is a concise per-class template, since UCSD provides only binary
labels. Acknowledge this in the paper's Section IV: "External ground truth is
a per-class template rather than image-specific free-text, so BLEU/ROUGE/BERTScore
measure linguistic alignment with the template; we additionally report binary
classification accuracy parsed from model responses."

Usage:
    python build_ucsd_test.py \\
        --covid_dir    /scratch/bhwr/msam1/external-eval/mosmed/https-github.com-UCSD-AI4H-COVID-CT/Images-processed/CT_COVID \\
        --noncovid_dir /scratch/bhwr/msam1/external-eval/mosmed/https-github.com-UCSD-AI4H-COVID-CT/Images-processed/CT_NonCOVID \\
        --output       /scratch/bhwr/msam1/external-eval/ucsd_test.json \\
        --n_per_class  100 \\
        --seed         42
"""
import argparse
import json
import os
import random


QUESTION = "Is this chest CT scan consistent with COVID-19 pneumonia? Describe the findings and provide your reasoning."

GT_COVID = (
    "This chest CT scan is consistent with COVID-19 pneumonia. Typical findings "
    "include bilateral, peripheral ground glass opacities with variable "
    "consolidation. These patterns support a diagnosis of COVID-19 viral pneumonia."
)

GT_NONCOVID = (
    "This chest CT scan is not consistent with COVID-19 pneumonia. "
    "The parenchyma does not show the characteristic bilateral peripheral "
    "ground glass opacities or multifocal consolidations seen in COVID-19."
)


def list_images(directory):
    """Return sorted image basenames (common CT image extensions)."""
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in exts
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--covid_dir", required=True)
    ap.add_argument("--noncovid_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_per_class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--staging_dir", default=None,
                    help="Optional. If set, images are COPIED into this flat "
                         "folder with class-prefixed names so evaluate_covid.py "
                         "can be pointed at one --image_folder")
    args = ap.parse_args()

    covid_imgs = list_images(args.covid_dir)
    noncovid_imgs = list_images(args.noncovid_dir)
    print(f"CT_COVID:    {len(covid_imgs)} files")
    print(f"CT_NonCOVID: {len(noncovid_imgs)} files")

    rng = random.Random(args.seed)
    if len(covid_imgs) < args.n_per_class or len(noncovid_imgs) < args.n_per_class:
        raise SystemExit(
            f"Not enough images: need {args.n_per_class} per class, "
            f"have COVID={len(covid_imgs)}, NonCOVID={len(noncovid_imgs)}"
        )

    covid_sample = rng.sample(covid_imgs, args.n_per_class)
    noncovid_sample = rng.sample(noncovid_imgs, args.n_per_class)

    samples = []
    for i, name in enumerate(covid_sample):
        samples.append({
            "id": f"ucsd_covid_{i:03d}",
            "image": f"covid_{name}",
            "src_path": os.path.join(args.covid_dir, name),
            "label": "COVID",
            "conversations": [
                {"from": "human", "value": QUESTION + "\n<image>"},
                {"from": "gpt", "value": GT_COVID},
            ],
        })
    for i, name in enumerate(noncovid_sample):
        samples.append({
            "id": f"ucsd_noncovid_{i:03d}",
            "image": f"noncovid_{name}",
            "src_path": os.path.join(args.noncovid_dir, name),
            "label": "NonCOVID",
            "conversations": [
                {"from": "human", "value": QUESTION + "\n<image>"},
                {"from": "gpt", "value": GT_NONCOVID},
            ],
        })

    rng.shuffle(samples)

    # Optional: copy images into one flat folder for evaluate_covid.py
    if args.staging_dir:
        import shutil
        os.makedirs(args.staging_dir, exist_ok=True)
        for s in samples:
            dst = os.path.join(args.staging_dir, s["image"])
            if not os.path.exists(dst):
                shutil.copy2(s["src_path"], dst)
        print(f"Staged {len(samples)} images into {args.staging_dir}")

    # Strip src_path before writing (it's not part of the schema)
    for s in samples:
        s.pop("src_path", None)

    with open(args.output, "w") as f:
        json.dump(samples, f, indent=2)

    n_cov = sum(1 for s in samples if s["label"] == "COVID")
    n_non = sum(1 for s in samples if s["label"] == "NonCOVID")
    print(f"\nWrote {len(samples)} samples ({n_cov} COVID + {n_non} NonCOVID) to {args.output}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
