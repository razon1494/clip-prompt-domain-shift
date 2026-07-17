"""End-to-end sanity test. Run once after install, before any experiments:
    python scripts/smoke_test.py

Checks:
1. Text-encoder parity: our TextEncoder (which injects continuous prompt
   embeddings) must reproduce clip_model.encode_text exactly when fed the
   embeddings of real tokens. This catches subtle divergences (e.g. a
   missing causal attention mask) that would otherwise corrupt results
   silently.
2. CoOp forward/backward: logits have the right shape, loss backprops, and
   gradients reach ONLY the learned context vectors.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.clip_model import CoOpCLIP, TextEncoder, load_clip

BACKBONE = "ViT-B-16"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"loading {BACKBONE} (downloads ~600MB on first run)...")
    clip_model, preprocess, tokenizer = load_clip(BACKBONE, device)

    # --- 1. parity check ---
    texts = ["a photo of a dog.", "a sketch of an elephant.", "an art painting of a guitar."]
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        ref = clip_model.encode_text(tokens)
        embeddings = clip_model.token_embedding(tokens)
        ours = TextEncoder(clip_model)(embeddings, tokens)
    max_diff = (ref - ours).abs().max().item()
    assert max_diff < 1e-4, f"FAIL: text encoder diverges from encode_text (max diff {max_diff:.2e})"
    print(f"PASS: text-encoder parity (max diff {max_diff:.2e})")

    # --- 2. CoOp forward/backward ---
    classnames = ["dog", "elephant", "guitar"]
    model = CoOpCLIP(classnames, clip_model, tokenizer, n_ctx=16, device=device).to(device)
    images = torch.randn(2, 3, 224, 224, device=device)
    logits = model(images)
    assert logits.shape == (2, 3), f"FAIL: logits shape {logits.shape}, expected (2, 3)"
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2], device=device))
    loss.backward()
    assert model.prompt_learner.ctx.grad is not None, "FAIL: no gradient on context vectors"
    frozen_grads = [p.grad for p in clip_model.parameters() if p.grad is not None]
    assert not frozen_grads, "FAIL: gradients leaked into the frozen CLIP backbone"
    print(f"PASS: CoOp forward/backward (loss {loss.item():.3f}, ctx grad norm "
          f"{model.prompt_learner.ctx.grad.norm().item():.4f})")

    n_trainable = sum(p.numel() for p in model.prompt_learner.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_trainable:,} (CoOp context vectors only)")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
