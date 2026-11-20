"""Local CUDA-graph benchmark for the four dense prefill projections.

This intentionally uses only public PyTorch operations so it runs on the local
SM86 GPU.  It is a tactic filter, not a substitute for H200 timing: the remote
submission uses SGLang's SM90 BF16 dispatcher.
"""

import argparse
import gc
import statistics

import torch
import torch.nn.functional as F


PROJECTIONS = {
    "qkv_gate": (2304, 8192),
    "o": (8192, 1024),
    "pooler": (4096, 8192),
    "unpooler": (8192, 4096),
}
DEFAULT_MS = (753, 3012, 16384)


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    torch.cuda.synchronize()
    return graph, output


def graph_median_us(graph, flush, repeats):
    samples = []
    for _ in range(repeats):
        flush.add_(1)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000.0)
    return statistics.median(samples), min(samples), max(samples)


def relative_l2(actual, expected):
    delta = actual.float() - expected.float()
    return (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float())).item()


def run_case(name, m, n, k, repeats, flush):
    torch.manual_seed(20261026 + m + n + k)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    weight_t = weight.t().contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    reference = F.linear(a, weight)
    torch.cuda.synchronize()

    variants = {
        "linear_rowmajor_weight": lambda: F.linear(a, weight),
        "mm_transpose_view": lambda: torch.mm(a, weight.t()),
        "mm_transposed_contiguous": lambda: torch.mm(a, weight_t),
        "mm_transposed_contiguous_out": lambda: torch.mm(a, weight_t, out=out),
        "addmm_transposed_out": lambda: torch.addmm(
            out, a, weight_t, beta=0.0, alpha=1.0, out=out
        ),
    }
    for chunks in (2, 4):
        bounds = tuple(i * m // chunks for i in range(chunks + 1))

        def chunked(bounds=bounds):
            for lo, hi in zip(bounds, bounds[1:]):
                torch.mm(a[lo:hi], weight_t, out=out[lo:hi])
            return out

        variants[f"mm_out_chunks{chunks}"] = chunked

    flops = 2 * m * n * k
    rows = []
    for variant, fn in variants.items():
        try:
            eager_output = fn()
            torch.cuda.synchronize()
            error = relative_l2(eager_output, reference)
            graph, graph_output = capture(fn)
            median_us, best_us, worst_us = graph_median_us(graph, flush, repeats)
            graph_error = relative_l2(graph_output, reference)
            tflops = flops / (median_us * 1.0e-6) / 1.0e12
            rows.append((median_us, variant, tflops, max(error, graph_error), best_us, worst_us))
            del graph, graph_output, eager_output
        except Exception as exc:
            print(
                f"ERROR op={name} M={m} variant={variant} "
                f"{type(exc).__name__}: {exc}"
            )
        gc.collect()
        torch.cuda.empty_cache()

    rows.sort()
    baseline = next(row for row in rows if row[1] == "linear_rowmajor_weight")
    print(f"CASE op={name} M={m} N={n} K={k} baseline_us={baseline[0]:.3f}")
    for median_us, variant, tflops, error, best_us, worst_us in rows:
        change = 100.0 * (median_us / baseline[0] - 1.0)
        print(
            f"  {variant:30s} median_us={median_us:10.3f} "
            f"change={change:+7.2f}% TFLOPS={tflops:8.3f} "
            f"rel_l2={error:.3e} range=[{best_us:.3f},{worst_us:.3f}]"
        )
    winner = rows[0]
    print(
        f"WINNER op={name} M={m} variant={winner[1]} "
        f"median_us={winner[0]:.3f} gain={100.0 * (baseline[0] / winner[0] - 1.0):.2f}%"
    )
    del a, weight, weight_t, out, reference
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="*", default=list(DEFAULT_MS))
    parser.add_argument("--ops", nargs="*", default=list(PROJECTIONS))
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA support is required")
    unknown = set(args.ops).difference(PROJECTIONS)
    if unknown:
        raise ValueError(f"unknown projections: {sorted(unknown)}")

    props = torch.cuda.get_device_properties(0)
    print(
        f"GPU name={props.name!r} capability={torch.cuda.get_device_capability(0)} "
        f"memory_GB={props.total_memory / 1e9:.3f} torch={torch.__version__}"
    )
    flush = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for m in args.m:
        for name in args.ops:
            n, k = PROJECTIONS[name]
            run_case(name, m, n, k, args.repeats, flush)


if __name__ == "__main__":
    main()
