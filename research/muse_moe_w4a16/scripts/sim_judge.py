import torch
import sys

def simulate_real_distribution(bs=32, n_trials=500):
    counts = []
    max_counts = []
    for seed in range(n_trials):
        gen = torch.Generator().manual_seed(seed)
        scores = torch.rand((bs, 256), generator=gen)
        z = torch.randn((256,), generator=gen)
        u = torch.rand((256,), generator=gen).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
        # topk
        topk_ids = torch.topk(scores, 8, dim=-1).indices # (bs, 8)
        # count for each GPU (8 GPUs, 32 experts each)
        gpu_counts = [0]*8
        for g in range(8):
            mask = (topk_ids >= g*32) & (topk_ids < (g+1)*32)
            gpu_counts[g] = mask.sum().item()
        # judge picks the busiest GPU
        busiest = max(range(8), key=lambda g: gpu_counts[g])
        lo = busiest * 32
        hi = lo + 32
        # count tokens per expert on this busiest GPU
        mask = (topk_ids >= lo) & (topk_ids < hi)
        local_ids = topk_ids[mask] - lo
        c = torch.bincount(local_ids, minlength=32)
        m = c.max().item()
        max_counts.append(m)
    print(f"M={bs} over {n_trials} real judge trials:")
    print(f"Max tokens on ANY expert: {max(max_counts)}")
    print(f"Avg max tokens on an expert: {sum(max_counts)/len(max_counts):.2f}")
    print(f"Trials with max > 16: {sum(1 for x in max_counts if x > 16)} / {n_trials}")
    print(f"Trials with max > 8: {sum(1 for x in max_counts if x > 8)} / {n_trials}")

simulate_real_distribution(bs=32, n_trials=500)
simulate_real_distribution(bs=16, n_trials=500)
