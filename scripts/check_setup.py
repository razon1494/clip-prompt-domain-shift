"""Sanity check: run this before anything else.
    python scripts/check_setup.py --data_root data/PACS
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.data import DomainDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None)
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    try:
        import open_clip

        print(f"open_clip version: {open_clip.__version__}")
    except ImportError:
        print("open_clip NOT installed — run: pip install -r requirements.txt")

    if args.data_root:
        try:
            ds = DomainDataset(args.data_root)
            print("\n" + ds.summary())
        except Exception as e:
            print(f"\nDataset check failed: {e}")


if __name__ == "__main__":
    main()
