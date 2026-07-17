import torch

# Deliberately style-diverse templates: PACS/OfficeHome domain shift is
# partly a style shift (photo/art/cartoon/sketch), so ensembling across
# styles gives zero-shot CLIP a fairer shot than a single template.
ZS_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "an image of a {}.",
    "a cropped photo of a {}.",
    "a drawing of a {}.",
    "a sketch of a {}.",
    "an art painting of a {}.",
]


@torch.no_grad()
def build_zeroshot_classifier(clip_model, tokenizer, classnames, templates, device):
    weights = []
    for name in classnames:
        texts = [t.format(name) for t in templates]
        tokens = tokenizer(texts).to(device)
        embeddings = clip_model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        weights.append(embeddings.mean(dim=0))
    weights = torch.stack(weights, dim=1)
    weights = weights / weights.norm(dim=0, keepdim=True)
    return weights.float()  # [dim, n_classes]
