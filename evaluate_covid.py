"""
Batch inference for COVID-FT LLaVA-Med on the CIDC held-out test set.
Key difference from evaluate_llava_med.py (Alz version):
  - Places <image> at END of question, matching the COVID training format
    where human turns are: "What do you see in this chest CT image?\\n<image>"

Produces predictions.json in the schema aggregate_covid_metrics.py reads:
    [{image, question, ground_truth, prediction}, ...]

Resumable (per-sample JSONL). Deterministic (do_sample=False).

Usage:
    python evaluate_covid.py \\
        --val_json   /work/hdd/bhwr/msam1/LLaVA-Med/data/alignment/llava_test.json \\
        --image_folder /work/hdd/bhwr/msam1/LLaVA-Med/images_covid/imagess \\
        --model_path /scratch/bhwr/msam1/llava-med-checkpoints/stage2-covid \\
        --output_dir ./eval_covid_seed42
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from transformers import AutoTokenizer
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


def load_model(model_path):
    disable_torch_init()
    print(f"[{time.strftime('%H:%M:%S')}] Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    print(f"[{time.strftime('%H:%M:%S')}] Loading model (fp16)...")
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda()

    model.config.mm_use_im_start_end = False
    model.config.mm_projector_type = "mlp2x_gelu"
    model.config.mm_vision_select_layer = -2
    model.config.mm_vision_tower = "openai/clip-vit-large-patch14-336"

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        print(f"[{time.strftime('%H:%M:%S')}] Loading vision tower...")
        vision_tower.load_model()
    vision_tower.to(device="cuda", dtype=torch.float16)
    image_processor = vision_tower.image_processor
    model.eval()
    print(f"[{time.strftime('%H:%M:%S')}] Model ready.")
    return tokenizer, model, image_processor


def extract_qa(sample):
    """Pull the first human turn (question text, no <image>) and first gpt turn."""
    convs = sample.get("conversations", [])
    if len(convs) < 2:
        return None, None
    human = convs[0]["value"].replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "").strip()
    gpt = convs[1]["value"].strip()
    return human, gpt


@torch.inference_mode()
def run_one(tokenizer, model, image_processor, pil_image, question, max_tokens):
    """Run the COVID-FT model on one (image, question) pair."""
    image_tensor = image_processor.preprocess(
        pil_image, return_tensors="pt"
    )["pixel_values"][0].unsqueeze(0).half().cuda()

    conv = conv_templates["vicuna_v1"].copy()
    # COVID training format: question first, <image> at end
    conv.append_message(conv.roles[0], question + "\n" + DEFAULT_IMAGE_TOKEN)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        temperature=1.0,
        max_new_tokens=max_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    return text


def load_done_indices(jsonl_path):
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "idx" in obj and "prediction" in obj:
                    done.add(obj["idx"])
            except json.JSONDecodeError:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_json", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", default="./eval_results")
    ap.add_argument("--max_tokens", type=int, default=400)   # matches COVID paper's judge config
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "predictions.jsonl")
    json_path = os.path.join(args.output_dir, "predictions.json")

    with open(args.val_json) as f:
        samples = json.load(f)
    if args.limit:
        samples = samples[: args.limit]
    print(f"Test samples: {len(samples)}")

    done = load_done_indices(jsonl_path)
    if done:
        print(f"Resuming: {len(done)} already done, {len(samples) - len(done)} remaining")

    tokenizer, model, image_processor = load_model(args.model_path)

    img_dir = Path(args.image_folder)
    out_f = open(jsonl_path, "a", buffering=1)
    stats = {"ok": 0, "img_missing": 0, "img_broken": 0, "model_err": 0, "schema_err": 0}

    try:
        for idx, sample in enumerate(tqdm(samples, desc="Predicting")):
            if idx in done:
                continue
            question, ground_truth = extract_qa(sample)
            if question is None or ground_truth is None:
                stats["schema_err"] += 1
                continue

            img_name = sample.get("image", "")
            img_path = img_dir / img_name
            if not img_path.exists():
                stats["img_missing"] += 1
                continue

            try:
                pil = Image.open(img_path).convert("RGB")
            except (UnidentifiedImageError, OSError):
                stats["img_broken"] += 1
                continue

            try:
                prediction = run_one(
                    tokenizer, model, image_processor, pil, question, args.max_tokens
                )
            except Exception as e:
                stats["model_err"] += 1
                out_f.write(json.dumps({
                    "idx": idx, "error": str(e)[:300], "image": img_name,
                }) + "\n")
                continue

            out_f.write(json.dumps({
                "idx": idx,
                "image": img_name,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "sample_id": sample.get("id", ""),
            }, ensure_ascii=False) + "\n")
            stats["ok"] += 1
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun with same args to resume.")
    finally:
        out_f.close()

    print(f"\nBuilding {json_path}...")
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "prediction" in obj:
                rows.append({
                    "image": obj["image"],
                    "question": obj["question"],
                    "ground_truth": obj["ground_truth"],
                    "prediction": obj["prediction"],
                })

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\nStats: {stats}")
    print(f"Wrote {len(rows)} predictions to {json_path}")


if __name__ == "__main__":
    main()
