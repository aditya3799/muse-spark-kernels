"""Physically remove dormant route shaping from the faster no-LT2 legal hybrid."""

import ast
from pathlib import Path


root = Path(__file__).resolve().parent.parent
base = (root / "submission_n72_legal_hybrid_candidate.py").read_text()
out = root / "submission_n72_exactroute_clean_candidate.py"


old = '''void k_moe_route(const float* logits, const float* bias, const int* lstart, int* ids, float* w, int* lid, int* pos, int* cnt,
                 int T, cudaStream_t st, int* tok_left, int* cq, int* cand, float* candsc, int* quota);'''
new = '''void k_moe_route(const float* logits, const float* bias, const int* lstart, int* ids, float* w, int* lid, int* pos, int* cnt,
                 int T, cudaStream_t st, int* tok_left, int* cq);'''
assert base.count(old) == 1
base = base.replace(old, new, 1)

start = base.index("void moe_route(torch::Tensor logits")
end = base.index("void moe_plan(", start)
wrapper = '''void moe_route(torch::Tensor logits, torch::Tensor bias, torch::Tensor lstart, torch::Tensor ids, torch::Tensor w,
               torch::Tensor lid, torch::Tensor pos, torch::Tensor cnt, torch::Tensor tok_left, torch::Tensor cq) {
  k_moe_route(logits.data_ptr<float>(), bias.data_ptr<float>(), lstart.data_ptr<int>(), ids.data_ptr<int>(), w.data_ptr<float>(),
              lid.data_ptr<int>(), pos.data_ptr<int>(), cnt.data_ptr<int>(), (int)logits.size(0), cur(),
              tok_left.numel() ? tok_left.data_ptr<int>() : nullptr, cq.numel() ? cq.data_ptr<int>() : nullptr);
}
'''
base = base[:start] + wrapper + base[end:]

start = base.index("void k_moe_route(const float* logits", base.index("GLUE = r'''"))
end = base.index("void k_moe_plan(", start)
glue = '''void k_moe_route(const float* logits, const float* bias, const int* lstart, int* ids, float* w, int* lid, int* pos, int* cnt,
                 int T, cudaStream_t st, int* tok_left, int* cq) {
  okc_launch(moe_route_kernel, dim3((T + ROUTE_WARPS - 1) / ROUTE_WARPS), dim3(ROUTE_WARPS * 32), 0, st,
             logits, bias, lstart, ids, w, lid, pos, cnt, T, tok_left, cq);
}
'''
base = base[:start] + glue + base[end:]

old = '''                                                                     int* __restrict__ pos, int* __restrict__ cnt, int T,
                                                                      int* __restrict__ tok_left = nullptr, int* __restrict__ cq = nullptr,
                                                                      int* __restrict__ cand = nullptr,
                                                                      float* __restrict__ candsc = nullptr,
                                                                      int* __restrict__ avail = nullptr) {'''
new = '''                                                                     int* __restrict__ pos, int* __restrict__ cnt, int T,
                                                                     int* __restrict__ tok_left = nullptr,
                                                                     int* __restrict__ cq = nullptr) {'''
assert base.count(old) == 1
base = base.replace(old, new, 1)

# Delete candidate-route selection inside the exact top-8 kernel.
rstart = base.index("__global__ void __launch_bounds__(ROUTE_WARPS * 32) moe_route_kernel")
start = base.index("    if (cand != nullptr) {", rstart)
end = base.index("    float sum = 0.f;", start)
base = base[:start] + base[end:]

# Delete all planner/apply/position-rebuild kernels for route manipulation.
start = base.index("// Conservative route shaping:")
end = base.index("// ---- combine:", start)
base = base[:start] + base[end:]

old = '''    shape_routes = ROUTE_SHAPING and T >= ROUTE_SHAPE_MIN_T
    core_zlen = LOCAL_EXPERTS + 1 + maxtiles + ((maxtiles + 4) if use_cit else 0)
    zlen = core_zlen + (96 if shape_routes else 0)  # counts | work | done | CIT state | route quotas/counters/availability
'''
new = '''    zlen = LOCAL_EXPERTS + 1 + maxtiles + ((maxtiles + 4) if use_cit else 0)
'''
assert base.count(old) == 1
base = base.replace(old, new, 1)

old = '''        cand = torch.empty((T,), dtype=torch.int32, device=dev) if shape_routes else st["no_zero"]
        candsc = torch.empty((T,), dtype=torch.float32, device=dev) if shape_routes else st["no_zero_f"]
        quota = zb[core_zlen:core_zlen + 96] if shape_routes else st["no_zero"]
        st["mod"].moe_route(logits, st["router_bias"], data["local_start"], ids, topw, lid, pos, cnt,
                            ctok[:T] if use_cit else st["no_zero"], ctok[T:2 * T] if use_cit else st["no_zero"],
                            cand, candsc, quota)  # one warp per token, then optional conservative route shaping
'''
new = '''        st["mod"].moe_route(logits, st["router_bias"], data["local_start"], ids, topw, lid, pos, cnt,
                            ctok[:T] if use_cit else st["no_zero"],
                            ctok[T:2 * T] if use_cit else st["no_zero"])  # exact top-8, one warp per token
'''
assert base.count(old) == 1
base = base.replace(old, new, 1)

for line in (
    'ROUTE_SHAPING = False  # exact top-8 routing: do not replace or drop near-tied local expert routes\n',
    'ROUTE_SHAPE_MIN_T = 512  # retained only for archived experiments; ignored while ROUTE_SHAPING is False\n',
):
    assert base.count(line) == 1
    base = base.replace(line, "", 1)

for forbidden in (
    "ROUTE_SHAPING",
    "ROUTE_SHAPE_MIN_T",
    "route_shape_",
    "candsc",
    "checker-optional",
    "base64.b64decode",
    "zlib.decompress",
):
    assert forbidden not in base, forbidden
assert "UNPOOL_FP8 = False" in base and "O_FP8 = False" in base
ast.parse(base)
compile(base, str(out), "exec")
out.write_bytes(base.encode())
print(out.name, len(base.encode()))
