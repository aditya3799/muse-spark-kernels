# okc: problem=muse_prefill_layer
"""Untimed H200 microbenchmark for the skinny BF16 O projection."""

import inspect

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flashinfer.fused_moe import cute_dsl_fused_moe_bf16
from nvfp4 import dequantize
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


MS = (753, 3012, 16384)
N = 8192
K = 1024
NUM_SMS = 132
RUN_BF16_MOE_GATE = False


@triton.jit
def _persistent_oproj_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    num_tiles,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    N_META: tl.constexpr,
    K_META: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS_META: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N_META, BLOCK_N)
    for tile_id in tl.range(pid, num_tiles, NUM_SMS_META):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id - pid_m * num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K_META, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_m[:, None] < m, other=0.0)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=offs_m[:, None] < m)


def _persistent_oproj(a, wt, out):
    block_m, block_n, block_k = 128, 256, 64
    num_tiles = triton.cdiv(a.shape[0], block_m) * triton.cdiv(N, block_n)
    grid = (min(NUM_SMS, num_tiles),)
    _persistent_oproj_kernel[grid](
        a,
        wt,
        out,
        a.shape[0],
        num_tiles,
        a.stride(0),
        a.stride(1),
        wt.stride(0),
        wt.stride(1),
        out.stride(0),
        out.stride(1),
        N_META=N,
        K_META=K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_SMS_META=grid[0],
        num_warps=8,
        num_stages=4,
    )
    return out


def _bench(name, fn, flops, warmup=4, repeats=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    times_us = sorted(start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends))
    median_us = times_us[len(times_us) // 2]
    tflops = flops / (median_us * 1e-6) / 1e12
    print(f"OPROJ M={flops // (2 * N * K)} variant={name} median_us={median_us:.3f} tflops={tflops:.3f}")
    return median_us


def prepare(weights):
    device = weights["o"].device
    w = weights["o"]
    wt_view = w.t()
    wt_contig = wt_view.contiguous()
    print(
        "OPROJ WEIGHT",
        f"shape={tuple(w.shape)} stride={w.stride()} contiguous={w.is_contiguous()}",
        f"aligned16={w.data_ptr() % 16 == 0}",
        f"wt_view_stride={wt_view.stride()} wt_view_contiguous={wt_view.is_contiguous()}",
        f"wt_contig_stride={wt_contig.stride()} wt_contig_aligned16={wt_contig.data_ptr() % 16 == 0}",
    )
    print("OPROJ torch._grouped_mm", hasattr(torch, "_grouped_mm"))
    if hasattr(torch, "_grouped_mm"):
        try:
            print("OPROJ grouped_doc", inspect.signature(torch._grouped_mm))
        except Exception as exc:
            print("OPROJ grouped_signature_error", type(exc).__name__, str(exc))

    for m in MS:
        a = torch.randn((m, K), dtype=torch.bfloat16, device=device)
        out = torch.empty((m, N), dtype=torch.bfloat16, device=device)
        print(
            "OPROJ INPUT",
            f"M={m} shape={tuple(a.shape)} stride={a.stride()} contiguous={a.is_contiguous()}",
            f"aligned16={a.data_ptr() % 16 == 0}",
            f"out_stride={out.stride()} out_contiguous={out.is_contiguous()}",
            f"out_aligned16={out.data_ptr() % 16 == 0}",
        )
        flops = 2 * m * N * K
        reference = _bf16_gemm_dispatch_impl(a, w, None)
        variants = {
            "nvjet_dispatch": lambda a=a: _bf16_gemm_dispatch_impl(a, w, None),
            "torch_mm_view": lambda a=a: torch.mm(a, wt_view),
            "torch_mm_contig_weight": lambda a=a: torch.mm(a, wt_contig),
            "torch_mm_contig_out": lambda a=a, out=out: torch.mm(a, wt_contig, out=out),
            "f_linear": lambda a=a: F.linear(a, w),
            "matmul_view": lambda a=a: torch.matmul(a, wt_view),
            "triton_persistent_128x256_wg2": lambda a=a, out=out: _persistent_oproj(
                a, wt_contig, out
            ),
        }
        for chunks in (2, 3, 4):
            bounds = [i * m // chunks for i in range(chunks + 1)]

            def chunked(a=a, out=out, bounds=bounds):
                for lo, hi in zip(bounds, bounds[1:]):
                    torch.mm(a[lo:hi], wt_contig, out=out[lo:hi])
                return out

            variants[f"torch_mm_out_chunks{chunks}"] = chunked

        for name, fn in variants.items():
            try:
                y = fn()
                max_abs = (y.float() - reference.float()).abs().max().item()
                print(f"OPROJ CHECK M={m} variant={name} max_abs={max_abs:.6g}")
                _bench(name, fn, flops)
            except Exception as exc:
                print(f"OPROJ ERROR M={m} variant={name} {type(exc).__name__}: {exc}")

        if hasattr(torch, "_grouped_mm"):
            attempts = (
                ("grouped_lists", lambda a=a: torch._grouped_mm([a], [wt_contig])),
                ("grouped_stacked", lambda a=a: torch._grouped_mm(a.unsqueeze(0), wt_contig.unsqueeze(0))),
            )
            for name, fn in attempts:
                try:
                    _bench(name, fn, flops)
                except Exception as exc:
                    print(f"OPROJ ERROR M={m} variant={name} {type(exc).__name__}: {exc}")

        del a, out, reference

    # Task 2's go/no-go gate: time only the BF16 routed GEMM pipeline.  Weight
    # expansion is untimed here, and the fused kernel performs W13, SwiGLU and
    # W2 in one call for the Q16K route distribution.
    if not RUN_BF16_MOE_GATE:
        raise RuntimeError("O projection microbenchmark complete")

    try:
        rows = torch.arange(4096, dtype=torch.long, device=device).view(-1, 32)
        row_map = torch.stack((rows + 4096, rows), dim=1).reshape(-1)
        w1 = dequantize(
            weights["w13"].index_select(1, row_map),
            weights["w13_sf"].index_select(1, row_map),
            weights["w13_gs"][:, None, None],
        ).to(torch.bfloat16)
        w2 = dequantize(
            weights["w2"], weights["w2_sf"], weights["w2_gs"][:, None, None]
        ).to(torch.bfloat16)
        tokens, top_k = 16384, 8
        x = torch.randn((tokens, 4096), dtype=torch.bfloat16, device=device)
        ids = (torch.arange(tokens * top_k, device=device) % 256).view(tokens, top_k).to(torch.int32)
        scales = torch.full((tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device)

        def bf16_moe():
            return cute_dsl_fused_moe_bf16(
                x,
                ids,
                scales,
                w1,
                w2,
                256,
                top_k,
                num_local_experts=32,
                local_expert_offset=0,
                use_fused_finalize=True,
                enable_pdl=True,
            )

        y = bf16_moe()
        print("BF16_MOE output", tuple(y.shape), y.dtype)
        _bench("bf16_fused_moe_only", bf16_moe, 1, warmup=1, repeats=6)
    except Exception as exc:
        print("BF16_MOE ERROR", type(exc).__name__, str(exc))

    raise RuntimeError("O projection and BF16 MoE microbenchmark complete")


def kernel_fn(data, state):
    raise RuntimeError("probe only")
