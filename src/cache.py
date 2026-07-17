from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import ImageListDataset

CACHE_DIR = Path("data/_feature_cache")


@torch.no_grad()
def domain_feature_map(dataset, domains, clip_model, preprocess, device, backbone, num_workers=4):
    """Compute (or load from disk) L2-normalized CLIP image features for every
    sample in the given domains. Returns {sample.path: feature_row (cpu)}.

    Valid because our preprocess is deterministic (resize + center-crop) and
    the image encoder is frozen: each image's feature is a constant, so
    caching changes nothing about training except wall-clock time.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feat_by_path = {}
    for domain in domains:
        samples = dataset.domain_samples(domain)
        key = f"{Path(dataset.root).name}_{backbone}_{domain}".replace(" ", "-")
        path = CACHE_DIR / f"{key}.pt"
        features = None
        if path.exists():
            blob = torch.load(path)
            if blob["features"].shape[0] == len(samples):
                features = blob["features"]
        if features is None:
            loader = DataLoader(
                ImageListDataset(samples, preprocess), batch_size=64, shuffle=False, num_workers=num_workers
            )
            chunks = []
            for images, _ in loader:
                f = clip_model.encode_image(images.to(device))
                f = f / f.norm(dim=-1, keepdim=True)
                chunks.append(f.cpu())
            features = torch.cat(chunks)
            torch.save({"features": features}, path)
        for s, row in zip(samples, features):
            feat_by_path[s.path] = row
    return feat_by_path


def feature_tensors(samples, feat_by_path):
    """Stack cached features for a sample list, preserving order."""
    feats = torch.stack([feat_by_path[s.path] for s in samples])
    labels = torch.tensor([s.class_idx for s in samples])
    return feats, labels
