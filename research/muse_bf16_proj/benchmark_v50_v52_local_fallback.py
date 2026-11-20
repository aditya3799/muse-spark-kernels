"""Compare v50, v52, and v53 Pooler fallback paths on the local CUDA GPU.

The persistent kernel is Hopper-only.  On pre-sm90 hardware v52 must use the
exact v50/cuBLAS fallback.  This verifies output parity and paired GPU timing;
it does not estimate H200 TMA/WGMMA performance.
"""

from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def timed(call) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1_000.0


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    v50 = load("local_v50", "submission_qkv32_o16_m16trace_v50.py")
    v52 = load("local_v52", "submission_pooler_persistent_v52.py")
    v53 = load("local_v53", "submission_pooler_persistent_sweep_v53.py")
    device = torch.device("cuda")
    torch.manual_seed(20261028)
    weight = torch.randn((4096, 8192), device=device, dtype=torch.bfloat16)
    flush = torch.empty(64 << 20, device=device, dtype=torch.uint8)
    state50 = {"lt": {}, "gv": {}, "pooler_raw": weight}
    state52 = {"lt": {}, "gv": {}, "pooler_raw": weight,
               "pwide": {"enabled": set()}}
    state53 = {"lt": {}, "gv": {}, "pooler_raw": weight,
               "pwide": {"enabled": set(), "selected": {}}}
    print("device", torch.cuda.get_device_name(), "capability", torch.cuda.get_device_capability())
    print("NOTE: sm90 persistent routes are disabled; this compares the fallback only.")
    for m in (128, 256):
        x = torch.randn((m, 8192), device=device, dtype=torch.bfloat16)
        data = {"config": {"layer": "pooler", "n": 4096}, "x": x}
        for _ in range(5):
            v50.kernel_fn(data, state50)
            v52.kernel_fn(data, state52)
            v53.kernel_fn(data, state53)
        torch.cuda.synchronize()
        a = v50.kernel_fn(data, state50)
        b = v52.kernel_fn(data, state52)
        c = v53.kernel_fn(data, state53)
        torch.cuda.synchronize()
        if not torch.equal(a, b) or not torch.equal(a, c):
            raise AssertionError(f"fallback mismatch for m={m}")

        samples50: list[float] = []
        samples52: list[float] = []
        samples53: list[float] = []
        for trial in range(21):
            flush.zero_()
            if trial & 1:
                samples52.append(timed(lambda: v52.kernel_fn(data, state52)))
                samples53.append(timed(lambda: v53.kernel_fn(data, state53)))
                samples50.append(timed(lambda: v50.kernel_fn(data, state50)))
            else:
                samples50.append(timed(lambda: v50.kernel_fn(data, state50)))
                samples53.append(timed(lambda: v53.kernel_fn(data, state53)))
                samples52.append(timed(lambda: v52.kernel_fn(data, state52)))
        med50 = statistics.median(samples50)
        med52 = statistics.median(samples52)
        med53 = statistics.median(samples53)
        ratios52 = [x / y for x, y in zip(samples52, samples50)]
        ratios53 = [x / y for x, y in zip(samples53, samples50)]
        print(f"m={m}: v50 {med50:.3f} us, v52-fallback {med52:.3f} us, "
              f"v53-fallback {med53:.3f} us, ratios "
              f"{statistics.median(ratios52):.5f}/{statistics.median(ratios53):.5f}, exact=True")


if __name__ == "__main__":
    main()
