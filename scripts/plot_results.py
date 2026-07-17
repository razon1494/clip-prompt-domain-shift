"""Per-domain accuracy chart with bootstrap CIs.
    python scripts/plot_results.py --results_dir results/PACS
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHOD_LABELS = {"zeroshot": "Zero-shot CLIP", "linear_probe": "Linear probe", "coop": "CoOp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    records = []
    for name in ["baseline_results.json", "coop_results.json"]:
        path = results_dir / name
        if path.exists():
            records += json.load(open(path))
    if not records:
        raise SystemExit(f"No result JSONs found in {results_dir}")

    by_method_domain = defaultdict(list)
    for r in records:
        by_method_domain[(r["method"], r["target_domain"])].append(r)

    domains = sorted({d for _, d in by_method_domain})
    methods = [m for m in METHOD_LABELS if any(m == mm for mm, _ in by_method_domain)]

    x = np.arange(len(domains))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(methods):
        means, err_lo, err_hi = [], [], []
        for d in domains:
            runs = by_method_domain.get((m, d), [])
            accs = [r["target_acc"] for r in runs]
            mean = np.mean(accs) if accs else np.nan
            means.append(mean * 100)
            # Single-seed methods carry the bootstrap CI; multi-seed uses seed spread.
            if len(accs) > 1:
                err_lo.append((mean - min(accs)) * 100)
                err_hi.append((max(accs) - mean) * 100)
            elif runs:
                err_lo.append((mean - runs[0]["ci_lo"]) * 100)
                err_hi.append((runs[0]["ci_hi"] - mean) * 100)
            else:
                err_lo.append(0)
                err_hi.append(0)
        ax.bar(x + i * width, means, width, yerr=[err_lo, err_hi], capsize=3, label=METHOD_LABELS[m])

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(domains)
    ax.set_ylabel("Held-out domain accuracy (%)")
    ax.set_title(f"Leave-one-domain-out accuracy — {results_dir.name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = results_dir / "per_domain_accuracy.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
