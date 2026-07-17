"""Combine baseline + CoOp results into one markdown table.
    python scripts/aggregate_results.py --results_dir results/PACS
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    baseline_path = results_dir / "baseline_results.json"
    coop_path = results_dir / "coop_results.json"
    baseline = json.load(open(baseline_path)) if baseline_path.exists() else []
    coop = json.load(open(coop_path)) if coop_path.exists() else []

    by_method_domain = defaultdict(list)
    for r in baseline + coop:
        by_method_domain[(r["method"], r["target_domain"])].append(r["target_acc"])

    domains = sorted({d for _, d in by_method_domain})
    method_order = ["zeroshot", "linear_probe", "coop"]
    methods = [m for m in method_order if any(m == mm for mm, _ in by_method_domain)]

    header = "| Method | " + " | ".join(domains) + " | Mean |"
    sep = "|" + "---|" * (len(domains) + 2)
    print(header)
    print(sep)
    for m in methods:
        row = [m]
        all_accs = []
        for d in domains:
            vals = by_method_domain.get((m, d), [])
            if vals:
                mean = np.mean(vals) * 100
                cell = f"{mean:.1f}±{np.std(vals) * 100:.1f}" if len(vals) > 1 else f"{mean:.1f}"
                row.append(cell)
                all_accs.extend(vals)
            else:
                row.append("-")
        row.append(f"{np.mean(all_accs) * 100:.1f}" if all_accs else "-")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
