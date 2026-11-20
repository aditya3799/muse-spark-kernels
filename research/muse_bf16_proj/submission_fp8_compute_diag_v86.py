# okc: problem=muse_bf16_proj
"""Diagnostic-only FP8-compute submission.

This deliberately changes the inner GEMM from BF16 to scaled E4M3 FP8 so that
we can measure real Hopper FP8 Tensor Core execution, including dynamic
activation conversion.  It is not a legal/accurate contest candidate: the
judge requires BF16 weights and activations (or a lossless representation).

The protected v50 submission is not imported or modified.
"""

from typing import TypedDict

import torch


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


FP8 = torch.float8_e4m3fn
FP8_HEADROOM = 224.0
# Probe activations are powers of two up to 128; 2x remains below E4M3's 448.
ACT_SCALE = 2.0


def _pack_weight(w: torch.Tensor):
    scale = FP8_HEADROOM / float(w.abs().max())
    # Keep the original [N,K] allocation row-major, then transpose it.  The
    # scaled_mm API requires its [K,N] right operand to be column-major.
    q_t = (w.float() * scale).to(FP8).t()
    inv = torch.tensor([1.0 / scale], dtype=torch.float32, device=w.device)
    return q_t, inv


def _fp8_mm(x: torch.Tensor, q_t: torch.Tensor, inv_x: torch.Tensor,
            inv_w: torch.Tensor, out_dtype: torch.dtype):
    xq = (x * ACT_SCALE).to(FP8)
    return torch._scaled_mm(
        xq, q_t, inv_x, inv_w, out_dtype=out_dtype, use_fast_accum=True
    )


def _capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _cold_graph_us(graph, flush, sink, rounds=7):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(rounds):
        torch.sum(flush, 0, out=sink)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def _diagnose(state, weights):
    # H200 has 50 MiB L2.  This 256 MiB buffer enforces a cold-weight run.
    dev = weights["qkv_gate"].device
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    print("FP8-COMPUTE diagnostic: scaled E4M3, conversion included", flush=True)
    for layer, m in (("qkv_gate", 1), ("o", 1), ("pooler", 1),
                     ("unpooler", 1), ("qkv_gate", 32),
                     ("o", 32), ("pooler", 32), ("unpooler", 32)):
        w = weights[layer]
        n = 1024 if layer == "unpooler" and m <= 8 else w.shape[0]
        x = torch.randn((m, w.shape[1]), dtype=torch.bfloat16, device=dev)
        wt = w[:n].t()
        q_t = state[layer + "_fp8_t"][:, :n]
        bf_graph = _capture(lambda: torch.mm(x, wt))
        fp_graph = _capture(lambda: _fp8_mm(
            x, q_t, state["_inv_x"], state[layer + "_inv_w"],
            torch.bfloat16
        ))
        bf_us = _cold_graph_us(bf_graph, flush, sink)
        fp_us = _cold_graph_us(fp_graph, flush, sink)
        print(
            "FP8-COMPUTE %s m%d: bf16 %.3f us fp8+convert %.3f us ratio %.4f" %
            (layer, m, bf_us, fp_us, fp_us / bf_us), flush=True
        )
        del bf_graph, fp_graph, x


def prepare(weights: Weights):
    state = {
        "_inv_x": torch.tensor(
            [1.0 / ACT_SCALE], dtype=torch.float32,
            device=weights["qkv_gate"].device
        )
    }
    for layer, w in weights.items():
        q_t, inv = _pack_weight(w)
        state[layer + "_fp8_t"] = q_t
        state[layer + "_inv_w"] = inv
    _diagnose(state, weights)
    return state


def kernel_fn(data: Inputs, state):
    layer = data["config"]["layer"]
    n = data["config"]["n"]
    out_dtype = torch.float32 if layer == "router" else torch.bfloat16
    return _fp8_mm(
        data["x"], state[layer + "_fp8_t"][:, :n], state["_inv_x"],
        state[layer + "_inv_w"], out_dtype
    )
