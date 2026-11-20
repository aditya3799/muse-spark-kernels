import torch

def gen_input(bs, seed):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen)
    z = torch.randn((256,), generator=gen)
    u = torch.rand((256,), generator=gen).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return ids

for bs in [1, 8, 16, 32, 128, 256]:
    local_experts_list = []
    total_tokens_list = []
    for seed in range(60):
        ids = gen_input(bs, 6443514289737657721 + seed * 1000)
        # local experts are 0..31
        local_ids = ids[(ids >= 0) & (ids < 32)]
        local_experts = len(torch.unique(local_ids))
        local_experts_list.append(local_experts)
        total_tokens_list.append(len(local_ids))
    avg_exp = sum(local_experts_list) / len(local_experts_list)
    avg_tok = sum(total_tokens_list) / len(total_tokens_list)
    print(f"M={bs:3d}: avg active local experts={avg_exp:5.1f} / 32, avg local tokens={avg_tok:5.1f}")
