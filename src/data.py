import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Sample:
    path: str
    class_idx: int
    domain_idx: int


class DomainDataset:
    """Expects root/<domain>/<class>/<image files>, e.g. PACS or OfficeHome
    after extraction. Domains and classes are discovered from the folder
    structure rather than hardcoded, so the same loader works for any dataset
    laid out this way.
    """

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.root}\n"
                f"See README.md 'Data setup' for download instructions."
            )
        self.domains = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        if not self.domains:
            raise RuntimeError(f"No domain subfolders found under {self.root}")

        class_set = None
        for domain in self.domains:
            classes = {d.name for d in (self.root / domain).iterdir() if d.is_dir()}
            class_set = classes if class_set is None else (class_set & classes)
        self.classes = sorted(class_set) if class_set else []
        if not self.classes:
            raise RuntimeError(
                f"No class folders shared across all domains under {self.root}. "
                f"Domains found: {self.domains}"
            )

        self.classnames = [c.replace("_", " ").replace("-", " ").strip().lower() for c in self.classes]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.domain_to_idx = {d: i for i, d in enumerate(self.domains)}

        self.samples = []
        for domain in self.domains:
            for cls in self.classes:
                folder = self.root / domain / cls
                if not folder.exists():
                    continue
                for f in folder.iterdir():
                    if f.suffix.lower() in IMG_EXTENSIONS:
                        self.samples.append(
                            Sample(str(f), self.class_to_idx[cls], self.domain_to_idx[domain])
                        )

    def summary(self):
        lines = [
            f"Root: {self.root}",
            f"Domains ({len(self.domains)}): {self.domains}",
            f"Classes ({len(self.classes)})",
            f"Total images: {len(self.samples)}",
        ]
        for domain in self.domains:
            di = self.domain_to_idx[domain]
            n = sum(1 for s in self.samples if s.domain_idx == di)
            lines.append(f"  {domain}: {n} images")
        return "\n".join(lines)

    def domain_samples(self, domain_name):
        di = self.domain_to_idx[domain_name]
        return [s for s in self.samples if s.domain_idx == di]


def split_train_val(samples, val_ratio=0.1, seed=0):
    """Stratified split by class so rare classes still appear in both splits."""
    rng = random.Random(seed)
    by_class = {}
    for s in samples:
        by_class.setdefault(s.class_idx, []).append(s)
    train, val = [], []
    for items in by_class.values():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val += items[:n_val]
        train += items[n_val:]
    return train, val


class ImageListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        return self.transform(img), s.class_idx
