import open_clip
import torch
import torch.nn as nn


def load_clip(backbone="ViT-B-16", device="cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained="openai")
    tokenizer = open_clip.get_tokenizer(backbone)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, tokenizer


class PromptLearner(nn.Module):
    """CoOp (Zhou et al., 2022): learns n_ctx continuous context vectors shared
    across classes, prepended to each class-name embedding before the frozen
    text transformer. Only self.ctx is trainable.
    """

    def __init__(self, classnames, clip_model, tokenizer, n_ctx=16, ctx_init=None, device="cuda"):
        super().__init__()
        n_cls = len(classnames)
        dtype = next(clip_model.parameters()).dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenizer([ctx_init]).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].clone()
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)

        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{prompt_prefix} {name}." for name in classnames]
        tokenized_prompts = tokenizer(prompts).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.tokenized_prompts = tokenized_prompts
        self.n_cls = n_cls
        self.n_ctx = n_ctx

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


class TextEncoder(nn.Module):
    """Replicates clip_model.encode_text's forward pass but starting from
    injected embeddings instead of a token-id lookup, since CoOp needs to
    feed continuous context vectors that have no discrete token id.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        # CLIP's text transformer is causal; omitting this mask silently
        # changes attention to bidirectional and corrupts text features.
        self.register_buffer("attn_mask", clip_model.attn_mask, persistent=False)

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(prompts.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(prompts.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CoOpCLIP(nn.Module):
    def __init__(self, classnames, clip_model, tokenizer, n_ctx=16, ctx_init=None, device="cuda"):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, tokenizer, n_ctx, ctx_init, device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.clip_model = clip_model
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

    def forward(self, images):
        # Image branch has no learnable parameters, so it is safe (and saves
        # memory) to keep it out of the autograd graph entirely.
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        return logit_scale * image_features.float() @ text_features.float().t()
