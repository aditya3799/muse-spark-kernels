"""Paired local A/B validation for dense projection tactics.

Baseline and candidate CUDA graphs alternate execution order.  Each replay is
preceded by an L2 flush, and the reported speedup is the median of paired
baseline/candidate ratios.
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

# Winners from bench_dense_local.py.  A baseline entry means that the first
# pass found no worthwhile alternative for that projection and shape.
TACTICS = {
    753: {
        "qkv_gate": "chunks2",
        "o": "chunks2",
        "pooler": "chunks2",
        "unpooler": "chunks2",
    },
    3012: {
        "qkv_gate": "baseline",
        "o": "transposed",
        "pooler": "chunks2",
        "unpooler": "transposed",
    },
    16384: {
        "qkv_gate": "transposed",
        "o": "chunks4",
        "pooler": "transposed",
        "unpooler": "transposed",
    },
}


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    torch.cuda.synchronize()
    return graph, output


def timed_replay(graph, flush):
    flush.add_(1)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000.0


def relative_l2(actual, expected):
    delta = actual.float() - expected.float()
    return (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float())).item()


def run_case(name, m, pairs, flush):
    n, k = PROJECTIONS[name]
    tactic = TACTICS[m][name]
    torch.manual_seed(20261026 + m + n + k)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    weight_t = weight.t().contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    baseline_fn = lambda: F.linear(a, weight)
    if tactic == "baseline":
        # Use an independently captured but mathematically identical view path
        # so the two CUDA graphs cannot share an allocation node.
        candidate_fn = lambda: torch.mm(a, weight.t())
    elif tactic == "transposed":
        candidate_fn = lambda: torch.mm(a, weight_t, out=out)
    elif tactic.startswith("chunks"):
        chunks = int(tactic.removeprefix("chunks"))
        bounds = tuple(i * m // chunks for i in range(chunks + 1))

        def candidate_fn():
            for lo, hi in zip(bounds, bounds[1:]):
                torch.mm(a[lo:hi], weight_t, out=out[lo:hi])
            return out
    else:
        raise ValueError(tactic)

    expected = baseline_fn()
    actual = candidate_fn()
    torch.cuda.synchronize()
    error = relative_l2(actual, expected)
    baseline_graph, _ = capture(baseline_fn)
    candidate_graph, candidate_output = capture(candidate_fn)
    graph_error = relative_l2(candidate_output, expected)

    baseline_times = []
    candidate_times = []
    ratios = []
    for pair in range(pairs):
        if pair % 2 == 0:
            baseline_us = timed_replay(baseline_graph, flush)
            candidate_us = timed_replay(candidate_graph, flush)
        else:
            candidate_us = timed_replay(candidate_graph, flush)
            baseline_us = timed_replay(baseline_graph, flush)
        baseline_times.append(baseline_us)
        candidate_times.append(candidate_us)
        ratios.append(baseline_us / candidate_us)

    median_baseline = statistics.median(baseline_times)
    median_candidate = statistics.median(candidate_times)
    paired_speedup = statistics.median(ratios)
    print(
        f"PAIRED op={name:8s} M={m:5d} tactic={tactic:10s} "
        f"baseline_us={median_baseline:10.3f} candidate_us={median_candidate:10.3f} "
        f"paired_speedup={paired_speedup:.5f} reduction={100.0 * (1.0 - 1.0 / paired_speedup):+.2f}% "
        f"rel_l2={max(error, graph_error):.3e}"
    )
    del (
        baseline_graph,
        candidate_graph,
        candidate_output,
        expected,
        actual,
        a,
        weight,
        weight_t,
        out,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return median_baseline, median_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="*", default=list(TACTICS))
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("a BF16-capable CUDA GPU is required")
    invalid = set(args.m).difference(TACTICS)
    if invalid:
        raise ValueError(f"unsupported M values: {sorted(invalid)}")

    props = torch.cuda.get_device_properties(0)
    print(
        f"GPU name={props.name!r} capability={torch.cuda.get_device_capability(0)} "
        f"memory_GB={props.total_memory / 1e9:.3f} torch={torch.__version__}"
    )
    flush = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for m in args.m:
        baseline_total = 0.0
        candidate_total = 0.0
        for name in PROJECTIONS:
            baseline_us, candidate_us = run_case(name, m, args.pairs, flush)
            baseline_total += baseline_us
            candidate_total += candidate_us
        print(
            f"TOTAL M={m} baseline_us={baseline_total:.3f} candidate_us={candidate_total:.3f} "
            f"speedup={baseline_total / candidate_total:.5f} "
            f"reduction={100.0 * (1.0 - candidate_total / baseline_total):+.2f}%"
        )


if __name__ == "__main__":
    main()
