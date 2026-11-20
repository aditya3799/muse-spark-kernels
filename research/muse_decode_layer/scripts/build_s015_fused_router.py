"""Build the isolated S015 M256 fused router-GEMM + exact routing candidate.

The active S010 submission remains untouched.  This script rewrites only the embedded
M256 routerwide source and the narrow Python dispatch points needed to consume routing
weights/IDs directly from that kernel.
"""

from __future__ import annotations

import ast
import base64
from pathlib import Path
import re
import zlib


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission.py"
TARGET = ROOT / "submission_s015_fused_router.py"


def _embedded_source(text: str, arm: int) -> tuple[str, str]:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "rw_SOURCES" for t in node.targets):
            continue
        assert isinstance(node.value, ast.Dict)
        for key, value in zip(node.value.keys, node.value.values):
            if ast.literal_eval(key) != arm:
                continue
            assert isinstance(value, ast.Call)
            blob = ast.literal_eval(value.args[0])
            decoded = zlib.decompress(base64.b64decode(blob)).decode()
            return blob, decoded
    raise RuntimeError(f"rw_SOURCES[{arm}] not found")


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def _fuse_cuda(src: str) -> str:
    helpers = r'''
struct __align__(16) rwdl_256_Cand { float key, raw; int id, pad; };
rwdl_256_RD float rwdl_256_sigmoid(float x) {
  return __frcp_rn(1.0f + __expf(-0.1767766952966369f * x));
}
rwdl_256_RD void rwdl_256_insert(rwdl_256_Cand a[8], rwdl_256_Cand v) {
  if(v.key <= a[7].key) return;
  a[7]=v;
#pragma unroll
  for(int j=6;j>=0;j--) if(a[j+1].key>a[j].key) { auto q=a[j];a[j]=a[j+1];a[j+1]=q; }
}
rwdl_256_RD rwdl_256_Cand rwdl_256_get(uint32_t base,int rank,int at) {
  float4 v=rwdl_256_pull(rwdl_256_map(base+at*16,rank));
  return {v.x,v.y,__float_as_int(v.z),0};
}
rwdl_256_RD void rwdl_256_put(unsigned char* sm,uint32_t base,int at,rwdl_256_Cand v) {
  *(float4*)(sm+(base-rwdl_256_smem(sm))+at*16)=make_float4(v.key,v.raw,__int_as_float(v.id),0.0f);
}
'''
    src = _replace_once(
        src,
        "rwdl_256_RD void rwdl_256_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }\n",
        "rwdl_256_RD void rwdl_256_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }\n" + helpers,
        "CUDA helpers",
    )
    src = _replace_once(
        src,
        "const __grid_constant__ CUtensorMap tx, float* out, int m)",
        "const __grid_constant__ CUtensorMap tx, const float* bias, float* route_w, int* route_id, int m)",
        "kernel signature",
    )
    src = _replace_once(
        src,
        "const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, rank=blockIdx.x%S;\n  const int row=blockIdx.x/S*NR, tok=blockIdx.y*TM;",
        "const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, rank=blockIdx.x%S, group=(blockIdx.x/S)&3;\n  const int row=group*NR, tok=blockIdx.y*TM;",
        "cluster coordinates",
    )

    begin = src.index("  auto emit=[&](int v,float4 z) {")
    end_marker = "\n  }\n}\n\ntypedef CUresult"
    end = src.index(end_marker, begin) + len("\n  }\n}")
    fused_tail = r'''  static_assert(BN==64 && S==4 && SW==0 && RED==2 && KC==1 && STACK==0);
  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
#pragma unroll
  for(int j=0;j<BN/8;j++) {
    const int v=j*128+tid, dst=v/OWN, local=v%OWN;
    const float4 val=make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);
    const uint32_t addr=recv+(rank*OWN+local)*16;
    const int peer=group*S+dst;
    if(dst==rank) *(float4*)(sm+(addr-base))=val;
    else rwdl_256_push(rwdl_256_map(addr,peer),val,rwdl_256_map(rb,peer));
  }
  asm volatile("bar.sync 1, 128;" :::"memory");
  rwdl_256_wait(rb,0);
  float* logits=(float*)sm;
  for(int t=tid;t<owned;t+=128) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int r=0;r<S;r++) rwdl_256_add(sum,*(float4*)(sm+recv-base+(r*OWN+t)*16));
    const int v=rank*OWN+t, ft=v%128, fj=v/128, fl=ft&31, fw=ft>>5;
    const int ar=fw*16+(fl>>2),bc=fj*8+(fl&3)*2;
    const float vv[4]={sum.x,sum.y,sum.z,sum.w};
#pragma unroll
    for(int c=0;c<4;c++) {
      const int tr=ar+(c/2)*8, ec=bc+(c&1)-rank*16;
      logits[tr*16+ec]=vv[c];
    }
  }
  asm volatile("bar.sync 1, 128;" :::"memory");
  const uint32_t cand_base=base+16384;
  const int token=tid>>1, half=tid&1, expert0=group*64+rank*16+half*8;
  rwdl_256_Cand best[8];
#pragma unroll
  for(int q=0;q<8;q++) {
    const float raw=rwdl_256_sigmoid(logits[token*16+half*8+q]);
    best[q]={raw+bias[expert0+q],raw,expert0+q,0};
  }
#pragma unroll
  for(int q=1;q<8;q++) for(int j=q;j>0;j--) if(best[j].key>best[j-1].key) { auto z=best[j];best[j]=best[j-1];best[j-1]=z; }
#pragma unroll
  for(int q=0;q<8;q++) {
    rwdl_256_Cand z={__shfl_xor_sync(0xffffffff,best[q].key,1),__shfl_xor_sync(0xffffffff,best[q].raw,1),__shfl_xor_sync(0xffffffff,best[q].id,1),0};
    if(!half) rwdl_256_insert(best,z);
  }
  if(!half) {
#pragma unroll
    for(int q=0;q<8;q++) rwdl_256_put(sm,cand_base,token*8+q,best[q]);
  }
  rwdl_256_sync();
  if(rank==0 && tid<64) {
#pragma unroll
    for(int q=0;q<8;q++) best[q]=rwdl_256_get(cand_base,group*S,tid*8+q);
#pragma unroll
    for(int r=1;r<S;r++) for(int q=0;q<8;q++) rwdl_256_insert(best,rwdl_256_get(cand_base,group*S+r,tid*8+q));
#pragma unroll
    for(int q=0;q<8;q++) rwdl_256_put(sm,cand_base,tid*8+q,best[q]);
  }
  rwdl_256_sync();
  if(group==0 && rank==0 && tid<64) {
#pragma unroll
    for(int q=0;q<8;q++) best[q]=rwdl_256_get(cand_base,0,tid*8+q);
#pragma unroll
    for(int g=1;g<4;g++) for(int q=0;q<8;q++) rwdl_256_insert(best,rwdl_256_get(cand_base,g*S,tid*8+q));
    float denom=0.0f;
#pragma unroll
    for(int q=0;q<8;q++) denom+=best[q].raw;
    const float inv=__frcp_rn(denom+1.0e-15f);
    const int tr=tok+tid;
    if(tr<m) {
#pragma unroll
      for(int q=0;q<8;q++) { route_w[tr*8+q]=best[q].raw*inv; route_id[tr*8+q]=best[q].id; }
    }
  }
  rwdl_256_sync();
}'''
    src = src[:begin] + fused_tail + src[end:]
    src = _replace_once(
        src,
        "int rwdl_256_launch_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query)",
        "int rwdl_256_launch_t(int64_t w,int64_t x,int64_t bias,int64_t route_w,int64_t route_id,int m,int64_t stream,int query)",
        "launch template signature",
    )
    src = _replace_once(
        src,
        "attr[0].val.clusterDim={S,1,1};",
        "attr[0].val.clusterDim={S*4,1,1};",
        "cluster dimension",
    )
    src = _replace_once(
        src,
        "cudaLaunchKernelEx(&cfg,fn,tw,tx,(float*)out,m)",
        "cudaLaunchKernelEx(&cfg,fn,tw,tx,(const float*)bias,(float*)route_w,(int*)route_id,m)",
        "kernel arguments",
    )
    src = _replace_once(
        src,
        "int rwdl_256_launch(int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query)",
        "int rwdl_256_launch(int id,int64_t w,int64_t x,int64_t bias,int64_t route_w,int64_t route_id,int m,int64_t stream,int query)",
        "export signature",
    )
    src = _replace_once(
        src,
        "(w,x,out,m,stream,query); default:return -99;}",
        "(w,x,bias,route_w,route_id,m,stream,query); default:return -99;}",
        "export call",
    )
    return src


def build() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    old_blob, cuda = _embedded_source(text, 256)
    fused = _fuse_cuda(cuda)
    new_blob = base64.b64encode(zlib.compress(fused.encode(), 9)).decode()
    text = _replace_once(text, old_blob, new_blob, "M256 source blob")

    old = """        cpp = '#include <stdint.h>\\n#include <pybind11/pybind11.h>\\nint ' + label + '_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);'"""
    new = """        if arm == 256:
            cpp = '#include <stdint.h>\\n#include <pybind11/pybind11.h>\\nint ' + label + '_launch(int,int64_t,int64_t,int64_t,int64_t,int64_t,int,int64_t,int);'
        else:
            cpp = '#include <stdint.h>\\n#include <pybind11/pybind11.h>\\nint ' + label + '_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);'"""
    text = _replace_once(text, old, new, "conditional C++ declaration")
    text = _replace_once(
        text,
        "state = {'enabled': True, 'weight': w, 'packed': packed, 'arms': {}}",
        "state = {'enabled': True, 'weight': w, 'packed': packed, 'bias': weights.get('router_bias'), 'arms': {}}",
        "routerwide bias state",
    )
    text = _replace_once(
        text,
        "        rw_run_routerwide_candidate(x, state, arm)",
        "        rw_run_routerwide_fused(x, state) if arm == 256 else rw_run_routerwide_candidate(x, state, arm)",
        "routerwide warmup",
    )
    anchor = """def rw_run_routerwide_candidate(x: torch.Tensor, state: dict, arm: int) -> torch.Tensor:
    out = torch.empty((x.shape[0], 256), device=x.device, dtype=torch.float32)
    rc = state['arms'][arm][1](0, state['packed'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream(x.device).cuda_stream, 0)
    if rc:
        raise RuntimeError(f'routerwide launch failed: {rc}')
    return out
"""
    addition = anchor + """

def rw_run_routerwide_fused(x: torch.Tensor, state: dict):
    route_w = torch.empty((x.shape[0], 8), device=x.device, dtype=torch.float32)
    route_id = torch.empty((x.shape[0], 8), device=x.device, dtype=torch.int32)
    bias = state['bias']
    rc = state['arms'][256][1](0, state['packed'].data_ptr(), x.data_ptr(), bias.data_ptr(), route_w.data_ptr(), route_id.data_ptr(), x.shape[0], torch.cuda.current_stream(x.device).cuda_stream, 0)
    if rc:
        raise RuntimeError(f'fused routerwide launch failed: {rc}')
    return route_w, route_id
"""
    text = _replace_once(text, anchor, addition, "fused Python launcher")
    text = _replace_once(
        text,
        "    if layer == 'router' and m in (128, 256):",
        "    if layer == 'router' and m == 128:",
        "disable generic M256 route",
    )
    text = _replace_once(
        text,
        "dense_weights = {name: weights[name] for name in (\"qkv_gate\", \"o\", \"pooler\", \"unpooler\", \"router\")}",
        "dense_weights = {name: weights[name] for name in (\"qkv_gate\", \"o\", \"pooler\", \"unpooler\", \"router\", \"router_bias\")}",
        "pass router bias",
    )
    text = _replace_once(
        text,
        """        logits = _e002_gemm(ffn_input, \"router\", N_EXPERTS, state)
        topk_weights, topk_ids = state[\"routing\"](None, logits, TOP_K, True)""",
        """        if tokens == 256:
            topk_weights, topk_ids = rw_run_routerwide_fused(ffn_input, state[\"dense_e002\"][\"bundle\"][\"rw\"])
        else:
            logits = _e002_gemm(ffn_input, \"router\", N_EXPERTS, state)
            topk_weights, topk_ids = state[\"routing\"](None, logits, TOP_K, True)""",
        "whole-layer fused dispatch",
    )
    if not text.startswith("# okc: problem=muse_decode_layer\n"):
        raise RuntimeError("problem marker changed")
    # submission.py is LF-only; avoid Windows' default CRLF expansion because the
    # challenge enforces a strict 1,000,000-byte source limit.
    with TARGET.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"wrote {TARGET.name}: {TARGET.stat().st_size} bytes; CUDA {len(cuda)} -> {len(fused)} bytes")


if __name__ == "__main__":
    build()
