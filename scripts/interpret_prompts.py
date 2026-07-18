"""Decode learned CoOp context vectors: nearest vocabulary tokens per position.
    python scripts/interpret_prompts.py --results_dir results/PACS
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.analysis import nearest_tokens
from src.clip_model import load_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--backbone", default="ViT-B-16")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, _ = load_clip(args.backbone, device)
    embedding_weight = clip_model.token_embedding.weight.detach().cpu()

    results_dir = Path(args.results_dir)
    out = {}
    for ctx_path in sorted(results_dir.glob("coop_ctx_*.pt")):
        ctx = torch.load(ctx_path)
        decoded = nearest_tokens(ctx, embedding_weight, topk=args.topk)
        out[ctx_path.stem] = decoded
        top1 = [d["tokens"][0] for d in decoded]
        print(f"{ctx_path.stem}:")
        print(f"  top-1 per position: {top1}")

    save_path = results_dir / "prompt_tokens.json"
    json.dump(out, open(save_path, "w"), indent=2)
    print(f"\nSaved {save_path}")


if __name__ == "__main__":
    main()
