"""Local SM86 validation for the Q16K grouped-Humming support kernels.

This intentionally does not emulate Humming's SM90 GEMMs.  It validates and
times the route permutation, compact SwiGLU, and compact fused post-norm/reduce
stages with CUDA graphs and an L2 flush between samples.
"""

import gc
import statistics

import torch
import triton
import triton.language as tl


TOP_K = 8
LOCAL_EXPERTS = 32
POST_NORM_EPS = 1e-8
ROW_PROGRAMS = 132


@triton.jit
def compact_silu_mul(
    gate_up, inter, expert_offsets,
    stride_gu_m, stride_gu_k, stride_out_m, stride_out_k,
    D: tl.constexpr, BLOCK_D: tl.constexpr, ROW_PROGRAMS: tl.constexpr,
    N_EXPERTS_LOCAL: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_rows = tl.load(expert_offsets + N_EXPERTS_LOCAL)
    while row < valid_rows:
        mask = cols < D
        gate = tl.load(gate_up + row * stride_gu_m + cols * stride_gu_k, mask=mask).to(tl.float32)
        up = tl.load(gate_up + row * stride_gu_m + (cols + D) * stride_gu_k, mask=mask).to(tl.float32)
        activated_gate = (gate * tl.sigmoid(gate)).to(tl.bfloat16)
        tl.store(
            inter + row * stride_out_m + cols * stride_out_k,
            activated_gate * up,
            mask=mask,
        )
        row += ROW_PROGRAMS


@triton.jit
def indexed_silu_mul(
    gate_up, inter, local_mask,
    stride_gu_m, stride_gu_k, stride_out_m, stride_out_k,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK_D + tl.arange(0, BLOCK_D)
    valid = tl.load(local_mask + row)
    mask = (cols < D) & valid
    gate = tl.load(gate_up + row * stride_gu_m + cols * stride_gu_k, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_up + row * stride_gu_m + (cols + D) * stride_gu_k, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        inter + row * stride_out_m + cols * stride_out_k,
        gate * tl.sigmoid(gate) * up,
        mask=mask,
    )


@triton.jit
def permute_rows(
    inputs, outputs, src2dst,
    num_tokens: tl.constexpr, D: tl.constexpr, TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    for start in tl.range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)
        mask = cols < D
        value = tl.load(inputs + token * D + cols, mask=mask)
        for route in tl.static_range(0, TOPK):
            dst = tl.load(src2dst + token * TOPK + route)
            if dst >= 0:
                tl.store(outputs + dst * D + cols, value, mask=mask)


@triton.jit
def compact_post_norm_reduce(
    down, src2dst, topk_weights, norm_weight, output,
    stride_down_m, stride_down_k, stride_tw_m, stride_tw_k,
    stride_out_m, stride_out_k,
    D: tl.constexpr, TOPK: tl.constexpr, BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    gain = tl.load(norm_weight + cols, mask=col_mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for route in tl.static_range(0, TOPK):
        dst = tl.load(src2dst + token * TOPK + route)
        valid = dst >= 0
        safe_dst = tl.where(valid, dst, 0)
        value = tl.load(
            down + safe_dst * stride_down_m + cols * stride_down_k,
            mask=col_mask & valid,
            other=0.0,
        ).to(tl.float32)
        scaled = (value * gain).to(tl.bfloat16).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(scaled * scaled, axis=0) / D + EPS)
        weight = tl.load(topk_weights + token * stride_tw_m + route * stride_tw_k).to(tl.float32)
        acc += tl.where(valid, scaled * inv_rms * weight, 0.0)
    tl.store(output + token * stride_out_m + cols * stride_out_k, acc, mask=col_mask)


@triton.jit
def indexed_post_norm_reduce(
    down, local_mask, topk_weights, norm_weight, output,
    stride_down_m, stride_down_k, stride_tw_m, stride_tw_k,
    stride_out_m, stride_out_k,
    D: tl.constexpr, TOPK: tl.constexpr, BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    gain = tl.load(norm_weight + cols, mask=col_mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for route in tl.static_range(0, TOPK):
        row = token * TOPK + route
        valid = tl.load(local_mask + row)
        value = tl.load(
            down + row * stride_down_m + cols * stride_down_k,
            mask=col_mask & valid,
            other=0.0,
        ).to(tl.float32)
        scaled = (value * gain).to(tl.bfloat16).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(scaled * scaled, axis=0) / D + EPS)
        weight = tl.load(topk_weights + token * stride_tw_m + route * stride_tw_k).to(tl.float32)
        acc += tl.where(valid, scaled * inv_rms * weight, 0.0)
    tl.store(output + token * stride_out_m + cols * stride_out_k, acc, mask=col_mask)


def make_routes(tokens: int, seed: int = 1234):
    gen = torch.Generator(device="cpu").manual_seed(seed + tokens)
    local = torch.rand(tokens * TOP_K, generator=gen) < (1.0 / 8.0)
    # Avoid the degenerate no-local-route case in tiny correctness tests.
    local[0] = True
    valid = torch.nonzero(local, as_tuple=False).flatten()
    src2dst = torch.full((tokens * TOP_K,), -1, dtype=torch.int32)
    src2dst[valid] = torch.arange(valid.numel(), dtype=torch.int32)
    offsets = torch.zeros(LOCAL_EXPERTS + 1, dtype=torch.int32)
    offsets[-1] = valid.numel()
    return local.cuda(), src2dst.cuda(), offsets.cuda(), int(valid.numel())


def launch_compact_activation(gate_up, inter, offsets, d):
    compact_silu_mul[(ROW_PROGRAMS, triton.cdiv(d, 256))](
        gate_up, inter, offsets,
        gate_up.stride(0), gate_up.stride(1), inter.stride(0), inter.stride(1),
        D=d, BLOCK_D=256, ROW_PROGRAMS=ROW_PROGRAMS,
        N_EXPERTS_LOCAL=LOCAL_EXPERTS, num_warps=4,
    )


def launch_compact_finalize(down, src2dst, weights, gain, output, d):
    compact_post_norm_reduce[(output.shape[0],)](
        down, src2dst, weights, gain, output,
        down.stride(0), down.stride(1), weights.stride(0), weights.stride(1),
        output.stride(0), output.stride(1), D=d, TOPK=TOP_K,
        BLOCK_D=triton.next_power_of_2(d), EPS=POST_NORM_EPS, num_warps=8,
    )


def correctness():
    tokens, d = 37, 256
    local, src2dst, offsets, valid_rows = make_routes(tokens)
    gate_up = torch.randn(tokens * TOP_K, 2 * d, device="cuda", dtype=torch.bfloat16)
    inter = torch.empty(tokens * TOP_K, d, device="cuda", dtype=torch.bfloat16)
    launch_compact_activation(gate_up, inter, offsets, d)
    ref_inter = torch.nn.functional.silu(gate_up[:valid_rows, :d]) * gate_up[:valid_rows, d:]
    torch.testing.assert_close(inter[:valid_rows], ref_inter, rtol=2e-2, atol=2e-2)

    down = torch.randn(tokens * TOP_K, d, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(tokens, TOP_K, device="cuda", dtype=torch.float32)
    gain = torch.randn(d, device="cuda", dtype=torch.bfloat16) * 0.1 + 1
    output = torch.empty(tokens, d, device="cuda", dtype=torch.bfloat16)
    launch_compact_finalize(down, src2dst, weights, gain, output, d)
    safe = src2dst.clamp_min(0).long().view(tokens, TOP_K)
    routed = down.index_select(0, safe.flatten()).view(tokens, TOP_K, d).float()
    scaled = (routed * gain.float()).to(torch.bfloat16).float()
    normalized = scaled * torch.rsqrt(scaled.square().mean(-1, keepdim=True) + POST_NORM_EPS)
    ref = (normalized * weights[..., None] * local.view(tokens, TOP_K, 1)).sum(1).to(torch.bfloat16)
    torch.testing.assert_close(output, ref, rtol=2e-2, atol=2e-2)
    print("correctness: PASS (compact SwiGLU and fused post-norm/reduce)")


def graph_median_us(fn, flush, repeats=21):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    samples = []
    for _ in range(repeats):
        flush.zero_()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def benchmark_shape(tokens: int, d: int, flush):
    routes = tokens * TOP_K
    local, src2dst, offsets, valid_rows = make_routes(tokens)
    result = {"tokens": tokens, "valid": valid_rows}

    x = torch.empty(tokens, d, device="cuda", dtype=torch.bfloat16)
    permuted = torch.empty(routes, d, device="cuda", dtype=torch.bfloat16)
    result["permute_us"] = graph_median_us(
        lambda: permute_rows[(tokens,)](
            x, permuted, src2dst, num_tokens=tokens, D=d, TOPK=TOP_K,
            BLOCK_D=512, num_warps=8,
        ),
        flush,
    )
    del x, permuted
    gc.collect()
    torch.cuda.empty_cache()

    gate_up = torch.empty(routes, 2 * d, device="cuda", dtype=torch.bfloat16)
    inter = torch.empty(routes, d, device="cuda", dtype=torch.bfloat16)
    result["indexed_act_us"] = graph_median_us(
        lambda: indexed_silu_mul[(routes, triton.cdiv(d, 256))](
            gate_up, inter, local,
            gate_up.stride(0), gate_up.stride(1), inter.stride(0), inter.stride(1),
            D=d, BLOCK_D=256, num_warps=4,
        ),
        flush,
    )
    result["compact_act_us"] = graph_median_us(
        lambda: launch_compact_activation(gate_up, inter, offsets, d), flush
    )
    del gate_up, inter
    gc.collect()
    torch.cuda.empty_cache()

    down = torch.empty(routes, d, device="cuda", dtype=torch.bfloat16)
    weights = torch.empty(tokens, TOP_K, device="cuda", dtype=torch.float32)
    gain = torch.empty(d, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(tokens, d, device="cuda", dtype=torch.bfloat16)
    result["indexed_finalize_us"] = graph_median_us(
        lambda: indexed_post_norm_reduce[(tokens,)](
            down, local, weights, gain, output,
            down.stride(0), down.stride(1), weights.stride(0), weights.stride(1),
            output.stride(0), output.stride(1), D=d, TOPK=TOP_K,
            BLOCK_D=triton.next_power_of_2(d), EPS=POST_NORM_EPS, num_warps=8,
        ),
        flush,
    )
    result["compact_finalize_us"] = graph_median_us(
        lambda: launch_compact_finalize(down, src2dst, weights, gain, output, d), flush
    )
    del down, weights, gain, output, local, src2dst, offsets
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}; SM {props.major}.{props.minor}; {props.total_memory / 2**30:.1f} GiB")
    correctness()
    flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    print("tokens  local_rows  permute_us  indexed_act  compact_act  act_speedup  indexed_final  compact_final")
    for tokens in (753, 3012, 16384):
        try:
            r = benchmark_shape(tokens, 4096, flush)
            speedup = r["indexed_act_us"] / r["compact_act_us"]
            print(
                f"{tokens:6d}  {r['valid']:10d}  {r['permute_us']:10.1f}  "
                f"{r['indexed_act_us']:11.1f}  {r['compact_act_us']:11.1f}  {speedup:11.2f}x  "
                f"{r['indexed_finalize_us']:13.1f}  {r['compact_finalize_us']:13.1f}"
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{tokens:6d}  SKIPPED (local GPU out of memory)")


if __name__ == "__main__":
    main()
