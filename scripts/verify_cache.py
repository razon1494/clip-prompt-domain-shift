"""Verify cached CoOp training is equivalent to uncached (same fold, same
seed, short run). Run once after enabling use_feature_cache:
    python scripts/verify_cache.py --config configs/pacs.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import yaml

from scripts.train_coop import train_one_fold
from src.data import DomainDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["epochs"] = args.epochs
    dataset = DomainDataset(cfg["data_root"])
    target = dataset.domains[0]
    out = Path("results/_verify")
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for cached in [True, False]:
        cfg["use_feature_cache"] = cached
        label = "CACHED" if cached else "UNCACHED"
        print(f"\n=== {label} run: target={target} seed=0 epochs={args.epochs} ===")
        r = train_one_fold(cfg, dataset, target, 0, "cuda", out)
        results[cached] = r
        print(f"  val={r['best_val_acc']:.4f}  target={r['target_acc']:.4f}")

    dv = abs(results[True]["best_val_acc"] - results[False]["best_val_acc"])
    dt = abs(results[True]["target_acc"] - results[False]["target_acc"])
    print(f"\nval diff={dv:.4f}  target diff={dt:.4f}")
    ok = dv < 0.01 and dt < 0.01
    print("EQUIVALENCE OK" if ok else "EQUIVALENCE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
