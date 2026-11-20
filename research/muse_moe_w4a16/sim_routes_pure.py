import random, math

def gen_input(bs, seed):
    rng = random.Random(seed)
    scores = [[rng.random() for _ in range(256)] for _ in range(bs)]
    # normal variates
    z = [rng.gauss(0, 1) for _ in range(256)]
    u = [max(1e-12, rng.random()) for _ in range(256)]
    g = [0.03 * z[i] / math.sqrt(-math.log(u[i])) for i in range(256)]
    for r in range(bs):
        for c in range(256):
            scores[r][c] += g[c]
    
    top8 = []
    gpu_bins = [0] * 8
    for r in range(bs):
        # find top 8 indices
        indexed = sorted(enumerate(scores[r]), key=lambda x: -x[1])[:8]
        row_ids = [idx for idx, val in indexed]
        top8.append(row_ids)
        for idx in row_ids:
            gpu_bins[idx // 32] += 1
            
    busiest = max(range(8), key=lambda i: gpu_bins[i])
    routed_ids = []
    for r in range(bs):
        routed_ids.extend([(idx - 32 * busiest) % 256 for idx in top8[r]])
    return routed_ids

for bs in [1, 8, 16, 32, 128, 256]:
    local_experts_list = []
    total_tokens_list = []
    max_tokens_per_expert = []
    for seed in range(30):
        ids = gen_input(bs, 12345 + seed * 1000)
        local_ids = [e for e in ids if 0 <= e < 32]
        unique_experts = set(local_ids)
        local_experts_list.append(len(unique_experts))
        total_tokens_list.append(len(local_ids))
        counts = [0] * 32
        for e in local_ids:
            counts[e] += 1
        max_tokens_per_expert.append(max(counts) if counts else 0)
    avg_exp = sum(local_experts_list) / len(local_experts_list)
    avg_tok = sum(total_tokens_list) / len(total_tokens_list)
    avg_max = sum(max_tokens_per_expert) / len(max_tokens_per_expert)
    print(f"M={bs:3d}: avg active local experts={avg_exp:5.1f}/32, avg local tokens={avg_tok:5.1f}, avg max tokens/expert={avg_max:4.1f}")
