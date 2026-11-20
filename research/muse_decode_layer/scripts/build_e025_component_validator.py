"""Extract E025's v49 backend into a nonzero-shard component validator."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
E025 = ROOT.parent / "scratch" / "muse_decode_e025" / "submission.py"
DESTINATION = ROOT.parent / "scratch" / "muse_moe_e025_validator" / "submission.py"


source = E025.read_text(encoding="utf-8")
start = source.index("# E024 embeds the independently validated BF16/NVFP4 v49 expert backend.")
start = source.index("\n", start) + 1
end = source.index("\ndef _marlin_bank(weights):", start)
component = source[start:end]

old_wrapper = '''def prepare(weights):
    return M_prepare(weights)
def kernel_fn(data, state):
    return M_kernel_fn(data, state)
'''
new_wrapper = '''def prepare(weights):
    state = M_prepare(weights)
    state["route_start_test"] = torch.full(
        (1,), 32, dtype=torch.int32, device=weights["w13"].device
    )
    return state
def kernel_fn(data, state):
    # The reference computes experts 0..31.  Shift every global ID to the next
    # EP shard, then have E025 localize with route_start=32 while deliberately
    # retaining packed weights 0..31.  Equal outputs prove route and weight
    # offsets are independent, including wraparound for nonlocal IDs.
    shifted = dict(data)
    shifted["topk_ids"] = (data["topk_ids"] + 32) % 256
    shifted["route_start"] = state["route_start_test"]
    shifted["config"] = dict(data["config"])
    shifted["config"]["local_start"] = 0
    return M_kernel_fn(shifted, state)
'''
count = component.count(old_wrapper)
if count != 1:
    raise RuntimeError(f"component wrapper: expected one match, found {count}")
component = component.replace(old_wrapper, new_wrapper, 1)

result = "# okc: problem=muse_moe_w4a16\n" + component
DESTINATION.parent.mkdir(parents=True, exist_ok=True)
DESTINATION.write_bytes(result.encode("utf-8"))
print(f"wrote {DESTINATION} ({len(result):,} bytes)")
