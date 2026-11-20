"""Generate a diagnostic-only H200 Codec E loader/decode probe from v105.

The generated submission leaves ``kernel_fn`` byte-for-byte routed through
v105.  Its added prepare-time probe proves the warp-interleaved pack exactly,
reports ptxas resources, and compares 1/2/4 Codec E substreams with the raw C12
high-code plane under alternating cold-L2 CUDA-graph timing.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_cgemv_threads_v105.py"
CODEC = ROOT / "codec_e.py"
PROBE = ROOT / "codec_e_cuda_probe.py"
OUT = ROOT / "submission_codec_e_probe_v147.py"


base = BASE.read_text()
codec = CODEC.read_text().replace("from __future__ import annotations\n", "")
probe = PROBE.read_text()
probe_start = probe.index('CUDA_SOURCE = r"""')
probe_end = probe.index("\ndef main():", probe_start)
probe_runtime = probe[probe_start:probe_end]


integration = r'''

# ---------------------------------------------------------------- v147: diagnostic-only lossless Codec E probe.
# No candidate path is reachable from kernel_fn.  This isolates the entropy
# loader/decode gate before any C12 MMA or escape-correction integration.
import statistics as _v147_statistics

_v147_base_prepare = prepare


def _v147_probe(weights):
    weight = weights["qkv_gate"]
    low, symbols, base, escapes, original = c12_pair_symbols(weight)
    del low, base, escapes, original
    stream_symbols = 256
    flat = symbols.reshape(-1)
    usable = flat.numel() // (32 * stream_symbols) * (32 * stream_symbols)
    symbols = flat[:usable].reshape(-1, 32, stream_symbols).contiguous()
    plan = build_plan(symbols)
    raw_input = symbols.view(symbols.shape[0], 32, -1, 16).permute(0, 2, 1, 3).contiguous()
    raw_output = torch.empty(symbols.shape[0] * 32, dtype=torch.int32, device=weight.device)
    ext = extension()
    raw_graph = _capture(lambda: ext.raw_hash(raw_input, raw_output))
    flush = torch.ones(64 * 1024 * 1024 // 1024, 1024,
                       dtype=torch.float32, device=weight.device)
    sink = torch.empty(1024, dtype=torch.float32, device=weight.device)
    raw_times = _cold_times(raw_graph, flush, sink, 13)
    raw_median = _v147_statistics.median(raw_times)
    print("PROBE ecod plan", plan.suffix_bits, plan.bases,
          "symbols", symbols.numel(), "raw_us", raw_median, flush=True)

    results = {}
    for substreams in (1, 2, 4):
        packed = pack_warp_interleaved(symbols, plan, substreams=substreams)
        reference_exact = bool(torch.equal(unpack_warp_interleaved(packed), symbols))
        output, suffix_bits, bases, lut = hash_gpu(packed)
        torch.cuda.synchronize(weight.device)
        device_exact = bool(torch.equal(output, raw_output))
        graph = _capture(
            lambda p=packed, o=output, sb=suffix_bits, b=bases, l=lut:
            ext.codec_e_hash(p.words, p.offsets.reshape(-1), o, sb, b, l,
                             p.symbols_per_stream, p.substreams))
        times = _cold_times(graph, flush, sink, 13)
        median = _v147_statistics.median(times)
        payload = 8.0 + packed.bits_per_symbol / 2.0
        allocated = 8.0 + (packed.allocated_bits / symbols.numel()) / 2.0
        info = int(ext.codec_e_hash_info(substreams))
        resources = {
            "registers": info & ((1 << 10) - 1),
            "local_bytes": (info >> 10) & ((1 << 20) - 1),
            "shared_bytes": (info >> 30) & ((1 << 20) - 1),
            "active_ctas_per_sm": info >> 50,
        }
        result = {
            "reference_exact": reference_exact,
            "device_exact": device_exact,
            "payload_bits_per_weight": payload,
            "allocated_bits_per_weight": allocated,
            "raw_us": raw_median,
            "codec_us": median,
            "ratio": median / raw_median,
            "resources": resources,
        }
        results[substreams] = result
        print("PROBE ecod", substreams, result, flush=True)
    return results


def prepare(weights):
    state = _v147_base_prepare(weights)
    weight = weights.get("qkv_gate")
    if (weight is None or not weight.is_cuda or
            torch.cuda.get_device_capability(weight.device) != (9, 0) or
            "H200" not in torch.cuda.get_device_name(weight.device)):
        return state
    try:
        state["_v147_codec_e_probe"] = _v147_probe(weights)
    except Exception as exc:
        print("PROBE ecod unavailable", type(exc).__name__, str(exc), flush=True)
    return state
'''


OUT.write_text(base + "\n\n" + codec + "\n\n" + probe_runtime + integration)
print(OUT)
