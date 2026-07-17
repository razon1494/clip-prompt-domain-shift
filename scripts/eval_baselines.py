"""Zero-shot CLIP and linear-probe baselines, leave-one-domain-out.
    python scripts/eval_baselines.py --config configs/pacs.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from src.baselines import ZS_TEMPLATES, build_zeroshot_classifier
from src.clip_model import load_clip
from src.data import DomainDataset, ImageListDataset
from src.metrics import bootstrap_ci, per_class_accuracy


@torch.no_grad()
def extract_features(clip_model, loader, device):
    feats, labels = [], []
    for images, y in loader:
        images = images.to(device)
        f = clip_model.encode_image(images)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = DomainDataset(cfg["data_root"])
    print(dataset.summary())

    clip_model, preprocess, tokenizer = load_clip(cfg["backbone"], device)
    zs_weights = build_zeroshot_classifier(clip_model, tokenizer, dataset.classnames, ZS_TEMPLATES, device)

    out_dir = Path(args.out) / cfg["dataset_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Feature extraction is shared across domains/methods — cache it.
    features_by_domain = {}
    for domain in dataset.domains:
        samples = dataset.domain_samples(domain)
        loader = DataLoader(
            ImageListDataset(samples, preprocess),
            batch_size=64,
            shuffle=False,
            num_workers=cfg["num_workers"],
            persistent_workers=cfg["num_workers"] > 0,
        )
        feats, labels = extract_features(clip_model, loader, device)
        features_by_domain[domain] = (feats, labels)
        print(f"  extracted features for {domain}: {feats.shape}")

    all_results = []
    for target_domain in dataset.domains:
        target_feats, target_labels = features_by_domain[target_domain]

        # Zero-shot
        logits = torch.from_numpy(target_feats).to(device) @ zs_weights
        preds = logits.argmax(dim=-1).cpu().numpy()
        correct = (preds == target_labels).astype(float)
        acc, lo, hi = bootstrap_ci(correct)
        all_results.append(
            {
                "method": "zeroshot",
                "target_domain": target_domain,
                "target_acc": acc,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_target": len(target_labels),
                "per_class_acc": per_class_accuracy(preds, target_labels, dataset.classes),
            }
        )

        # Linear probe: fit on all source domains, test on held-out target
        source_domains = [d for d in dataset.domains if d != target_domain]
        src_feats = np.concatenate([features_by_domain[d][0] for d in source_domains])
        src_labels = np.concatenate([features_by_domain[d][1] for d in source_domains])

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(src_feats, src_labels)
        preds = clf.predict(target_feats)
        correct = (preds == target_labels).astype(float)
        acc, lo, hi = bootstrap_ci(correct)
        all_results.append(
            {
                "method": "linear_probe",
                "target_domain": target_domain,
                "target_acc": acc,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_target": len(target_labels),
                "per_class_acc": per_class_accuracy(preds, target_labels, dataset.classes),
            }
        )
        print(f"{target_domain}: zeroshot={all_results[-2]['target_acc']:.3f}  linear_probe={acc:.3f}")

    json.dump(all_results, open(out_dir / "baseline_results.json", "w"), indent=2)
    print(f"\nSaved to {out_dir / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
