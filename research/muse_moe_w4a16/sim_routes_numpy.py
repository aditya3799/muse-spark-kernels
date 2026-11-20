import numpy as np

def gen_input(bs, seed):
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    scores = rng.rand(bs, 256)
    z = rng.randn(256)
    u = np.clip(rng.rand(256), 1e-12, 1.0)
    scores = scores + 0.03 * z / np.sqrt(-np.log(u))
    # top 8 per token
    top8 = np.argpartition(-scores, 8, axis=-1)[:, :8]
    # busiest GPU relabel
    gpu_bins = np.bincount((top8 // 32).flatten(), minlength=8)
    busiest = gpu_bins.argmax()
    ids = (top8 - 32 * busiest) % 256
    return ids

for bs in [1, 8, 16, 32, 128, 256]:
    local_experts_list = []
    total_tokens_list = []
    max_tokens_per_expert = []
    for seed in range(60):
        ids = gen_input(bs, 12345 + seed * 1000)
        local_ids = ids[(ids >= 0) & (ids < 32)]
        local_experts = len(np.unique(local_ids))
        local_experts_list.append(local_experts)
        total_tokens_list.append(len(local_ids))
        if len(local_ids) > 0:
            counts = np.bincount(local_ids, minlength=32)
            max_tokens_per_expert.append(counts.max())
        else:
            max_tokens_per_expert.append(0)
    avg_exp = sum(local_experts_list) / len(local_experts_list)
    avg_tok = sum(total_tokens_list) / len(total_tokens_list)
    avg_max = sum(max_tokens_per_expert) / len(max_tokens_per_expert)
    print(f"M={bs:3d}: avg active local experts={avg_exp:5.1f}/32, avg local tokens={avg_tok:5.1f}, avg max tokens/expert={avg_max:4.1f}")
