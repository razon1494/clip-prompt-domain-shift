# Do Learned Prompts Generalize? Auditing CLIP Prompt Learning under Domain Shift

Prompt learning (CoOp) adapts CLIP to a downstream task by training a few thousand
parameters instead of fine-tuning the backbone — but prompts learned on one visual
domain are known to overfit its style. This project audits that failure mode with a
controlled comparison of **zero-shot CLIP**, **prompt ensembles**, **linear probing**,
and **CoOp** under leave-one-domain-out evaluation on **PACS** and **OfficeHome**,
asking not just *how much* accuracy drops under domain shift, but *where* (per-domain
and per-class slices) and *why* (interpretation of the learned context vectors).

Motivated by work on generalizable prompt learning: CoOp/CoCoOp (Zhou et al.),
Style-Pro (WACV 2025), and DiSa (SIGKDD 2025).

## Methods compared

| Method | Trainable params | What it isolates |
|---|---|---|
| Zero-shot CLIP (single template) | 0 | Pretrained representation alone |
| Zero-shot + prompt ensemble | 0 | Effect of hand-written style-diverse prompts |
| Linear probe | ~0.3K–33K | Supervised head on frozen features |
| CoOp (16 context vectors) | ~8K | Learned continuous prompts |

## Protocol

- **Leave-one-domain-out (LODO):** train on all source domains, evaluate on the held-out
  target domain. Model selection uses a validation split of the *source* domains only —
  the target domain is never touched before final evaluation.
- **3 seeds** per (method, target-domain) cell for trained methods.
- **1000-resample bootstrap 95% CIs** on per-example correctness.
- **Slice analysis:** per-domain and per-class accuracy, to show what aggregate numbers hide.
- **Prompt interpretation:** nearest vocabulary tokens to each learned context vector.

## Setup

```bash
pip install -r requirements.txt
python scripts/smoke_test.py   # verifies CLIP + CoOp implementation end-to-end
```

### Data

Place datasets under `data/` as `data/<DATASET>/<domain>/<class>/<images>`:

```
data/PACS/art_painting/dog/*.jpg        # 4 domains, 7 classes, ~10k images
data/OfficeHome/Art/Alarm_Clock/*.jpg   # 4 domains, 65 classes, ~15.5k images
```

- **PACS:** download via [DomainBed](https://github.com/facebookresearch/DomainBed) or a
  Kaggle mirror (search "PACS domain generalization").
- **OfficeHome:** download from the [official page](https://www.hemanthdv.org/officeHomeDataset.html).

Domain/class folders are discovered automatically. Verify with:

```bash
python scripts/check_setup.py --data_root data/PACS
```

## Run

```bash
python scripts/eval_baselines.py --config configs/pacs.yaml        # zero-shot + linear probe
python scripts/train_coop.py     --config configs/pacs.yaml        # CoOp, all folds x seeds (resumable)
python scripts/aggregate_results.py --results_dir results/PACS     # markdown results table
python scripts/plot_results.py      --results_dir results/PACS     # per-domain chart
```

Same commands with `configs/officehome.yaml` for OfficeHome. All experiments run on a
single 6GB consumer GPU (CLIP backbone frozen throughout; only context vectors train).

## Results

*(In progress — results tables, per-domain charts, slice analysis, and learned-prompt
token analysis will be added as runs complete.)*

## References

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021
- Zhou et al., *Learning to Prompt for Vision-Language Models* (CoOp), IJCV 2022
- Zhou et al., *Conditional Prompt Learning for Vision-Language Models* (CoCoOp), CVPR 2022
- Talemi et al., *Style-Pro: Style-Guided Prompt Learning for Generalizable Vision-Language Models*, WACV 2025
- Talemi et al., *DiSa: Directional Saliency-Aware Prompt Learning for Generalizable Vision-Language Models*, SIGKDD 2025
- Li et al., *Deeper, Broader and Artier Domain Generalization* (PACS), ICCV 2017
- Venkateswara et al., *Deep Hashing Network for Unsupervised Domain Adaptation* (OfficeHome), CVPR 2017
