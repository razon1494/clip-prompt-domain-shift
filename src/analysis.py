import torch


def nearest_tokens(ctx_vectors, token_embedding_weight, topk=5):
    """For each learned CoOp context position, find the topk closest real
    vocabulary tokens by cosine similarity. Useful to sanity-check whether
    the learned prompt drifted toward interpretable words or into
    uninterpretable space.

    token_embedding_weight: clip_model.token_embedding.weight, shape [vocab, dim]
    """
    from open_clip.tokenizer import SimpleTokenizer

    tokenizer = SimpleTokenizer()
    ctx = ctx_vectors.float()
    vocab = token_embedding_weight.float()
    sims = torch.nn.functional.cosine_similarity(
        ctx.unsqueeze(1), vocab.unsqueeze(0), dim=-1
    )  # [n_ctx, vocab_size]

    results = []
    for i in range(ctx.shape[0]):
        top = sims[i].topk(topk)
        tokens = [tokenizer.decoder.get(idx.item(), "?").replace("</w>", "") for idx in top.indices]
        results.append(
            {"position": i, "tokens": tokens, "scores": [round(s, 3) for s in top.values.tolist()]}
        )
    return results
