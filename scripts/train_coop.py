"""CoOp training, leave-one-domain-out, multi-seed.
    python scripts/train_coop.py --config configs/pacs.yaml
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.cache import domain_feature_map, feature_tensors
from src.clip_model import CoOpCLIP, load_clip
from src.data import DomainDataset, ImageListDataset, split_train_val
from src.metrics import bootstrap_ci, per_class_accuracy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def train_one_fold(cfg, dataset, target_domain, seed, device, out_dir):
    use_cache = cfg.get("use_feature_cache", False)
    if use_cache:
        # Feature extraction consumes no torch RNG, so doing it before
        # set_seed/model init keeps runs comparable with the uncached path.
        clip_model, preprocess, tokenizer = load_clip(cfg["backbone"], device)
        feat_by_path = domain_feature_map(
            dataset, dataset.domains, clip_model, preprocess, device, cfg["backbone"], cfg["num_workers"]
        )

    set_seed(seed)
    if not use_cache:
        clip_model, preprocess, tokenizer = load_clip(cfg["backbone"], device)

    source_domains = [d for d in dataset.domains if d != target_domain]
    source_samples = []
    for d in source_domains:
        source_samples += dataset.domain_samples(d)
    train_samples, val_samples = split_train_val(source_samples, cfg["val_ratio"], seed)
    target_samples = dataset.domain_samples(target_domain)

    if use_cache:
        train_loader = DataLoader(
            TensorDataset(*feature_tensors(train_samples, feat_by_path)),
            batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(*feature_tensors(val_samples, feat_by_path)), batch_size=cfg["batch_size"]
        )
        target_loader = DataLoader(
            TensorDataset(*feature_tensors(target_samples, feat_by_path)), batch_size=cfg["batch_size"]
        )
    else:
        common = dict(num_workers=cfg["num_workers"], persistent_workers=cfg["num_workers"] > 0)
        train_loader = DataLoader(
            ImageListDataset(train_samples, preprocess), batch_size=cfg["batch_size"], shuffle=True, drop_last=True, **common
        )
        val_loader = DataLoader(ImageListDataset(val_samples, preprocess), batch_size=cfg["batch_size"], shuffle=False, **common)
        target_loader = DataLoader(
            ImageListDataset(target_samples, preprocess), batch_size=cfg["batch_size"], shuffle=False, **common
        )

    model = CoOpCLIP(
        dataset.classnames, clip_model, tokenizer, cfg["n_ctx"], cfg.get("ctx_init"), device
    ).to(device)

    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=cfg["lr"], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc, best_state = -1.0, None
    for epoch in range(cfg["epochs"]):
        model.train()
        model.clip_model.eval()  # frozen backbone: keep it in eval mode (BN/dropout) even while model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        preds, val_labels = evaluate(model, val_loader, device)
        val_acc = float((preds == val_labels).mean())
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.prompt_learner.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == cfg["epochs"] - 1:
            print(f"    epoch {epoch+1}/{cfg['epochs']}  val_acc={val_acc:.3f}  best={best_val_acc:.3f}")

    model.prompt_learner.load_state_dict(best_state)
    preds, target_labels = evaluate(model, target_loader, device)
    correct = (preds == target_labels).astype(float)
    acc, lo, hi = bootstrap_ci(correct)

    result = {
        "method": "coop",
        "target_domain": target_domain,
        "seed": seed,
        "best_val_acc": best_val_acc,
        "target_acc": acc,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_target": len(target_samples),
        "per_class_acc": per_class_accuracy(preds, target_labels, dataset.classes),
    }

    ctx_path = out_dir / f"coop_ctx_{target_domain}_seed{seed}.pt"
    torch.save(model.prompt_learner.ctx.detach().cpu(), ctx_path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = DomainDataset(cfg["data_root"])
    print(dataset.summary())

    out_dir = Path(args.out) / cfg["dataset_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "coop_results.json"

    all_results = json.load(open(results_path)) if results_path.exists() else []
    done = {(r["target_domain"], r["seed"]) for r in all_results}

    for target_domain in dataset.domains:
        for seed in cfg["seeds"]:
            if (target_domain, seed) in done:
                print(f"skip target={target_domain} seed={seed} (already done)")
                continue
            print(f"\n=== target={target_domain} seed={seed} ===")
            result = train_one_fold(cfg, dataset, target_domain, seed, device, out_dir)
            print(f"  -> target_acc={result['target_acc']:.3f} [{result['ci_lo']:.3f}, {result['ci_hi']:.3f}]")
            all_results.append(result)
            json.dump(all_results, open(results_path, "w"), indent=2)

    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
