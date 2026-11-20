import torch
import reference

dev = "cpu"
bs = 32
for seed in [1234, 5252, 9999, 42, 777]:
    inp = reference.generate_input(bs=bs, sharding="ep8", seed=seed, device=dev)
    ids = inp["topk_ids"].flatten()
    local = ids[(ids >= 0) & (ids < 32)]
    counts = torch.bincount(local, minlength=32)
    touched = (counts > 0).sum().item()
    total_local = local.numel()
    max_c = counts.max().item()
    print(f"seed={seed}: total_local_pairs={total_local}/256 ({total_local/256*100:.1f}%), touched_experts={touched}/32, max_tokens_per_expert={max_c}")
