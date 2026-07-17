import numpy as np


def accuracy(preds, labels):
    return float((preds == labels).mean())


def bootstrap_ci(correct, n_boot=1000, seed=0, alpha=0.05):
    """correct: 1D array of 0/1 per-example correctness."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    boot_accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_accs[i] = correct[idx].mean()
    lo = np.percentile(boot_accs, 100 * alpha / 2)
    hi = np.percentile(boot_accs, 100 * (1 - alpha / 2))
    return float(correct.mean()), float(lo), float(hi)


def per_class_accuracy(preds, labels, class_names):
    out = {}
    for c, name in enumerate(class_names):
        mask = labels == c
        if mask.sum() > 0:
            out[name] = float((preds[mask] == labels[mask]).mean())
    return out
