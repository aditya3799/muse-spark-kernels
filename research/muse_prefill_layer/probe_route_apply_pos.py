"""Local correctness and paired timing probe for the fused route apply/position pass.

Only the exact routing section and launch wrapper are compiled.  The laptop is SM86,
whereas unrelated kernels in the full submission intentionally require Hopper/SM89+
instructions and cannot be assembled locally.
"""

from __future__ import annotations

import importlib
import hashlib
import statistics

import torch
from torch.utils.cpp_extension import load_inline


ROUTE_CPP = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
void k_moe_route(const float* logits, const float* bias, const int* lstart, int* ids, float* w,
                 int* lid, int* pos, int* cnt, int T, cudaStream_t st, int* tok_left, int* cq,
                 int* cand, float* candsc, int* quota);
static cudaStream_t cur() { return c10::cuda::getCurrentCUDAStream(); }
void moe_route(torch::Tensor logits, torch::Tensor bias, torch::Tensor lstart, torch::Tensor ids,
               torch::Tensor w, torch::Tensor lid, torch::Tensor pos, torch::Tensor cnt,
               torch::Tensor tok_left, torch::Tensor cq, torch::Tensor cand,
               torch::Tensor candsc, torch::Tensor quota) {
  k_moe_route(logits.data_ptr<float>(), bias.data_ptr<float>(), lstart.data_ptr<int>(),
              ids.data_ptr<int>(), w.data_ptr<float>(), lid.data_ptr<int>(), pos.data_ptr<int>(),
              cnt.data_ptr<int>(), (int)logits.size(0), cur(),
              tok_left.numel() ? tok_left.data_ptr<int>() : nullptr,
              cq.numel() ? cq.data_ptr<int>() : nullptr,
              cand.numel() ? cand.data_ptr<int>() : nullptr,
              candsc.numel() ? candsc.data_ptr<float>() : nullptr,
              quota.numel() ? quota.data_ptr<int>() : nullptr);
}
'''


def build_local(module_name: str):
    mod = importlib.import_module(module_name)
    cap = torch.cuda.get_device_capability()
    arch = f"{cap[0]}{cap[1]}"
    start = mod._CUDA_SRC.index("// ---- routing:")
    end = mod._CUDA_SRC.index("// ---- combine:", start)
    route = mod._CUDA_SRC[start:end]
    gstart = mod.GLUE.index("void k_moe_route")
    gend = mod.GLUE.index("void k_moe_plan", gstart)
    launch = mod.GLUE[gstart:gend]
    src = r'''
#define OKC_PDL 0
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define DEVI __device__ __forceinline__
__device__ __forceinline__ void pdl_enter() {}
namespace okc_moe {
''' + route + r'''
}  // namespace okc_moe
using namespace okc_moe;
template <typename... KArgs, typename... Args>
static inline void okc_launch(void (*k)(KArgs...), dim3 g, dim3 b, size_t smem,
                              cudaStream_t st, Args... args) {
  k<<<g, b, smem, st>>>(((KArgs)args)...);
}
''' + launch
    return load_inline(
        f"{module_name}_route_probe_{hashlib.sha1(src.encode()).hexdigest()[:10]}",
        cpp_sources=ROUTE_CPP,
        cuda_sources=src,
        functions=["moe_route"],
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "--use_fast_math"],
    )


def make_case(t: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn((t, 512), generator=gen, device="cuda", dtype=torch.float32) * 1.75
    bias = torch.randn((256,), generator=gen, device="cuda", dtype=torch.float32) * 0.08
    lstart = torch.tensor([(17 * seed + 29) & 255], device="cuda", dtype=torch.int32)
    return logits, bias, lstart


def padded_rows(cnt: torch.Tensor) -> int:
    vals = cnt.to(torch.int64)
    q, r = vals // 256, vals % 256
    pad = torch.where(r == 0, 0,
          torch.where(r <= 32, 32,
          torch.where(r <= 64, 64,
          torch.where(r <= 128, 128,
          torch.where(r <= 160, 160,
          torch.where(r <= 192, 192, 256))))))
    return int((q * 256 + pad).sum())


def run_route(mod, case, shaped: bool):
    logits, bias, lstart = case
    t = logits.shape[0]
    ids = torch.empty((t, 8), device="cuda", dtype=torch.int32)
    weights = torch.empty((t, 8), device="cuda", dtype=torch.float32)
    lid = torch.empty((t, 8), device="cuda", dtype=torch.int32)
    pos = torch.empty((t, 8), device="cuda", dtype=torch.int32)
    cnt = torch.zeros((32,), device="cuda", dtype=torch.int32)
    empty_i = torch.empty((0,), device="cuda", dtype=torch.int32)
    empty_f = torch.empty((0,), device="cuda", dtype=torch.float32)
    cand = torch.empty((t,), device="cuda", dtype=torch.int32) if shaped else empty_i
    candsc = torch.empty((t,), device="cuda", dtype=torch.float32) if shaped else empty_f
    quota = torch.zeros((96,), device="cuda", dtype=torch.int32) if shaped else empty_i
    mod.moe_route(logits, bias, lstart, ids, weights, lid, pos, cnt, empty_i, empty_i, cand, candsc, quota)
    torch.cuda.synchronize()
    return ids, weights, lid, pos, cnt, cand, candsc, quota


def validate_case(mod, t: int, seed: int):
    case = make_case(t, seed)
    ref = run_route(mod, case, False)
    got = run_route(mod, case, True)
    ids0, _, lid0, _, cnt0, *_ = ref
    ids, weights, lid, pos, cnt, _, _, quota = got

    if not torch.allclose(weights.sum(1), torch.ones(t, device="cuda"), atol=3e-7, rtol=0):
        raise AssertionError("route weights are not normalized")
    if (ids.sort(1).values[:, 1:] == ids.sort(1).values[:, :-1]).any():
        raise AssertionError("duplicate expert in a token's top-8")
    ls = int(case[2].item())
    expected_lid = (ids - ls) & 255
    expected_lid = torch.where(expected_lid < 32, expected_lid, -1)
    if not torch.equal(lid, expected_lid):
        raise AssertionError("ids/lid mismatch")
    final_cnt = torch.bincount(lid[lid >= 0].to(torch.int64), minlength=32).to(torch.int32)
    if not torch.equal(cnt, final_cnt):
        raise AssertionError("final count mismatch")
    for expert in range(32):
        ep = pos[lid == expert].sort().values
        if not torch.equal(ep, torch.arange(ep.numel(), device="cuda", dtype=torch.int32)):
            raise AssertionError(f"positions for local expert {expert} are not dense and unique")

    changed_slots = ids != ids0
    per_token = changed_slots.sum(1)
    if int(per_token.max()) > 1:
        raise AssertionError("more than one route changed for a token")
    changed = int(changed_slots.sum())
    initial_local = int(cnt0.sum())
    if changed > initial_local * 7 // 100:
        raise AssertionError("planner exceeded the 7% local-route budget")
    if changed and not bool((lid0[changed_slots] >= 0).all() and (lid[changed_slots] < 0).all()):
        raise AssertionError("a shaped route was not local-to-nonlocal")
    planned = quota[:32].to(torch.int64)
    attempted = quota[32:64].to(torch.int64)
    fulfilled = torch.minimum(planned, attempted)
    shortfall = torch.clamp(planned - attempted, min=0)
    return (changed, initial_local, int(planned.sum()), int(fulfilled.sum()), int(shortfall.sum()),
            padded_rows(cnt0), padded_rows(cnt))


def time_route(mod, case, rounds: int = 41):
    t = case[0].shape[0]
    outputs = run_route(mod, case, True)
    ids, weights, lid, pos, cnt, cand, candsc, quota = outputs
    empty_i = torch.empty((0,), device="cuda", dtype=torch.int32)
    flush = torch.empty((64 << 20,), device="cuda", dtype=torch.uint8)
    samples = []
    for i in range(rounds + 8):
        cnt.zero_()
        flush.fill_(i)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        mod.moe_route(case[0], case[1], case[2], ids, weights, lid, pos, cnt,
                      empty_i, empty_i, cand, candsc, quota)
        end.record()
        end.synchronize()
        if i >= 8:
            samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples), statistics.quantiles(samples, n=4)[0], statistics.quantiles(samples, n=4)[2]


def main():
    print("building candidate", flush=True)
    candidate = build_local("submission_route_avail_candidate")
    for t in (512, 753, 3012, 16384):
        for seed in (3, 19):
            changed, local, planned, fulfilled, shortfall, pad0, pad1 = validate_case(candidate, t, seed)
            print(f"candidate invariant T={t} seed={seed}: changed={changed}/{local} "
                  f"planned={planned} fulfilled={fulfilled} shortfall={shortfall} "
                  f"padded={pad0}->{pad1} saved={pad0-pad1}", flush=True)

    print("building scored control", flush=True)
    control = build_local("submission_scored_910fa5d559e44d0d")
    case = make_case(16384, 101)
    c0_cnt = run_route(candidate, case, False)[4]
    b0_cnt = run_route(control, case, False)[4]
    print(f"unshaped count vectors equal: {torch.equal(c0_cnt, b0_cnt)}", flush=True)
    c_cnt = run_route(candidate, case, True)[4]
    b_cnt = run_route(control, case, True)[4]
    diff = c_cnt - b_cnt
    print(f"shaped count vectors equal: {torch.equal(c_cnt, b_cnt)}; max_abs_diff={int(diff.abs().max())}; "
          f"total_candidate={int(c_cnt.sum())}; total_control={int(b_cnt.sum())}", flush=True)
    if not torch.equal(c_cnt, b_cnt):
        print("candidate counts:", c_cnt.cpu().tolist(), flush=True)
        print("control counts:  ", b_cnt.cpu().tolist(), flush=True)
    print(f"seed=101 padded candidate={padded_rows(c_cnt)} control={padded_rows(b_cnt)} "
          f"saved={padded_rows(b_cnt)-padded_rows(c_cnt)}", flush=True)
    for t in (753, 3012, 16384):
        for seed in (37, 101, 211):
            probe_case = make_case(t, seed)
            cc = run_route(candidate, probe_case, True)[4]
            bc = run_route(control, probe_case, True)[4]
            print(f"work T={t} seed={seed}: candidate={padded_rows(cc)} control={padded_rows(bc)} "
                  f"saved={padded_rows(bc)-padded_rows(cc)} local_delta={int(bc.sum()-cc.sum())}", flush=True)
    for label, module in (("control", control), ("candidate", candidate), ("control", control), ("candidate", candidate)):
        med, q1, q3 = time_route(module, case)
        print(f"{label}: median={med:.2f} us, IQR=[{q1:.2f}, {q3:.2f}]", flush=True)


if __name__ == "__main__":
    main()
