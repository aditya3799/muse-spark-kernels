"""Generate v149: two row-parallel warpgroups over 128 C12 weight rows.

The protected v105 file is preserved byte-for-byte as the prefix.  The new
kernel is a structural follow-up to v115: two warpgroups own disjoint 64-row
halves of one 128-row output block and share one TMA activation ring.  K is
split only across CTAs, giving the one-wave grids described by the pw34 plan.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_cgemv_threads_v105.py"
V55 = ROOT / "submission_unpooler128_c12_v55.py"
OUT = ROOT / "submission_r128_rowwg_v149.py"

base_bytes = BASE.read_bytes()
base = base_bytes.decode()
old = V55.read_text()

start = old.index("# ---------------------------------------------------------------- v32: lossless C12 unpooler-32")
end = old.index("\n_u32_mod = [None]", start)
section = old[start:end]
section = section.replace("_q128_zsrc(", "_v149_zsrc(")
section = section.replace(
    "ext = _u32_ext() if instances == _u32_INSTANCES else _u32_build(instances)",
    "ext = _u32_build(instances)",
)
section = section.replace(
    "_u32_CUDA_CORE + _u32_CUDA_MAP",
    "_v149_core() + _u32_CUDA_MAP",
)
section = section.replace(
    "(_u32_mma(n) for n in (16, 32, 64))",
    "(_u32_mma(n) for n in (16, 32, 64, 128, 256))",
)
section = section.replace(
    "_u32_CUDA_HOST.replace('DISPATCH'",
    "_v149_host().replace('DISPATCH'",
)
section = section.replace("for c in (inst[5], False):", "for c in (inst[5],):")
section = section.replace(
    "for compressed in (inst[5], False):", "for compressed in (inst[5],):"
)

prelude = r'''
import base64 as _v149_base64
import hashlib
import zlib as _v149_zlib


def _v149_zsrc(payload):
    return _v149_zlib.decompress(_v149_base64.b64decode(payload)).decode()
'''

integration = r'''

# ---------------------------------------------------------------- v149: R128 row-parallel register-A WGMMA.
# Unlike v115, the two warpgroups own separate 64-row halves and share the
# activation TMA ring.  Cross-CTA K split is doubled to keep the same one-wave
# grids while halving activation traffic per CTA.
import base64 as _v149_base64
import hashlib as _v149_hashlib
import zlib as _v149_zlib


def _v149_zsrc(payload):
    return _v149_zlib.decompress(_v149_base64.b64decode(payload)).decode()


def _v149_once(src, old, new):
    if src.count(old) != 1:
        raise RuntimeError("v149 source anchor mismatch: " + old[:80])
    return src.replace(old, new, 1)


def _v149_core():
    src = _u32_CUDA_CORE
    src = _v149_once(
        src,
        "constexpr int RW=R/16,NT=128*WK,RP=R+4,SB=BN*128,WS=(Q*(SB+8)+1023)/1024*1024,NL=C12?3:4;",
        "constexpr int RW=R/16,NT=128*WK,RP=R+4,SB=BN*128,NL=C12?3:4;",
    )
    src = _v149_once(
        src,
        "int tid=threadIdx.x,wg=tid/128,warp=tid%128/32,lane=tid%32;",
        "int tid=threadIdx.x,wg=tid/128,warp=tid%128/32,lane=tid%32,grow=wg*4+warp;",
    )
    src = _v149_once(
        src,
        "uint32_t base0=addr(sm),base=base0+wg*WS,full=base+Q*SB;",
        "uint32_t base0=addr(sm),base=base0,full=base+Q*SB,empty=full+Q*8;",
    )
    src = _v149_once(
        src,
        "int kb0=(k/64)*(rank*WK+wg)/(split*WK),nkb=(k/64)*(rank*WK+wg+1)/(split*WK)-kb0;",
        "int kb0=(k/64)*rank/split,nkb=(k/64)*(rank+1)/split-kb0;",
    )
    if src.count("tid%128==0") != 3:
        raise RuntimeError("v149 expected three warpgroup-leader anchors")
    src = src.replace("tid%128==0", "tid==0")
    src = _v149_once(
        src,
        'for(int s=0;s<Q;s++)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");',
        'for(int s=0;s<Q;s++){asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory"); asm volatile("mbarrier.init.shared::cta.b64 [%0],2;" :: "r"(empty+s*8) : "memory");}',
    )
    src = _v149_once(src, "if(C12 && warp<RW) {", "if(C12 && grow<RW) {")
    src = _v149_once(
        src,
        "((nt*split*WK+rank*WK+wg)*RW+warp)*64;",
        "((nt*split+rank)*RW+grow)*64;",
    )
    src = _v149_once(src, "if(i<nkb && warp<RW) {", "if(i<nkb && grow<RW) {")
    src = _v149_once(src, "*RW+warp)*NL*32+lane;", "*RW+grow)*NL*32+lane;")
    src = _v149_once(
        src,
        "      loadx(i+Q);",
        "      if(warp==0 && lane==0)arrive(empty+s*8);\n"
        "      if(tid==0 && i+Q<nkb){waitbar(empty+s*8,(i/Q)&1);loadx(i+Q);}",
    )
    src = _v149_once(
        src,
        'for(int s=0;s<Q;s++)asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(full+s*8) : "memory");',
        'for(int s=0;s<Q;s++){asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(full+s*8) : "memory"); asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(empty+s*8) : "memory");}',
    )
    src = _v149_once(
        src,
        "int row=warp*16+lane/4,col=2*(lane%4);",
        "int row=grow*16+lane/4,col=2*(lane%4);",
    )
    src = _v149_once(
        src,
        "((float*)sm)[(wg*BN+col+8*j)*RP+row+8*h]=acc[4*j+2*h];",
        "((float*)sm)[(col+8*j)*RP+row+8*h]=acc[4*j+2*h];",
    )
    src = _v149_once(
        src,
        "((float*)sm)[(wg*BN+col+8*j+1)*RP+row+8*h]=acc[4*j+2*h+1];",
        "((float*)sm)[(col+8*j+1)*RP+row+8*h]=acc[4*j+2*h+1];",
    )
    src = _v149_once(
        src,
        "for(int pp=0;pp<split*WK;pp++) {\n      int s=pp/WK;\n      uint32_t remote=base0+((pp%WK*BN+token)*RP+row)*4;",
        "for(int pp=0;pp<split;pp++) {\n      int s=pp;\n      uint32_t remote=base0+(token*RP+row)*4;",
    )
    src = _v149_once(
        src,
        "if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));",
        "if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));\n"
        "      if constexpr(BN==128)mma128(acc,a,desc(base+s*SB));\n"
        "      if constexpr(BN==256)mma256(acc,a,desc(base+s*SB));",
    )
    return src


def _v149_host():
    return _u32_CUDA_HOST.replace(
        "constexpr int a=WK*((Q*(BN*128+8)+1023)/1024*1024),b=WK*BN*(R+4)*4,smem=a>b?a:b;",
        "constexpr int a=Q*BN*128+Q*16,b=BN*(R+4)*4,smem=a>b?a:b;",
    )


def _v149_pack_weight(w, r, wk, split, compressed=True):
    n, k = w.shape
    total = split
    rw = r // 16
    if w.dtype != torch.bfloat16 or r != 128 or n % r or k % 64 or (k // 64 < total):
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(
        0, 3, 2, 6, 4, 5, 1, 7
    ).reshape(n // 16, k // 64, 32, 32)
    esc = torch.full((n // r * total * rw, 64), -1, dtype=torch.int32, device=w.device)
    table, max_esc, escapes = (0, 0, 0)
    if compressed:
        data, b, entries = _u32_sm_c12_bytes(v)
        if not torch.equal(_u32_sm_c12_unpack(data, b, entries), v):
            return None
        table = _u32_sm_table(b) & 18446744073709551615
        er, ec, el, ev, xor7 = entries
        escapes = er.numel()
        if escapes:
            boundaries = torch.tensor(
                [k // 64 * s // total for s in range(1, total)], device=w.device
            )
            rank = torch.bucketize(ec, boundaries, right=True)
            seq = ec - k // 64 * rank // total
            group = (er // rw * total + rank) * rw + er % rw
            max_esc = int(torch.bincount(group, minlength=esc.shape[0]).max())
            if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
                return None
            order = torch.argsort(group * (1 << 14) + seq)
            gs = group[order]
            ss = seq[order]
            pos = torch.arange(len(gs), device=w.device) - torch.searchsorted(gs, gs)
            entry = (
                ss.long() << 18
                | el[order].long() << 13
                | (ev[order] // 2).long() << 8
                | (ev[order] % 2).long() << 7
                | xor7[order].long()
            )
            esc[gs, pos] = _u32__to_i32(entry)
    else:
        data = _u32_sm_bf16_bytes(v)
    payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
    return dict(
        w=payload, esc=esc, table=table, r=r, wk=wk, split=split,
        compressed=compressed, n=n, k=k, escapes=escapes,
        max_esc=max_esc, pack_exact=True,
    )


_u32_pack_weight = _v149_pack_weight

_V149_CONFIGS = {}
for _v149_m in (128, 256):
    _V149_CONFIGS[_v149_m, "qkv_gate"] = ((_v149_m, 128, 2, 2, 2, True, 1), 8)
    _V149_CONFIGS[_v149_m, "o"] = ((_v149_m, 128, 2, 2, 2, True, 1), 2)
    _V149_CONFIGS[_v149_m, "pooler"] = ((_v149_m, 128, 2, 2, 2, True, 1), 4)
    _V149_CONFIGS[_v149_m, "unpooler"] = ((_v149_m, 128, 2, 2, 2, True, 1), 2)

_V149_REPLAY_US = 4.0
_V149_ROUTE_RATIO = 0.97
_V149_SCORE_GATE = 0.009
_V149_FLUSH = [None, None]


def _v149_time(fn, flush, repeats=13):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            fn()
    if _V149_FLUSH[0] is not flush:
        _V149_FLUSH[0] = flush
        _V149_FLUSH[1] = torch.empty(1024, dtype=torch.float32, device=flush.device)
    samples = []
    for _ in range(repeats):
        torch.sum(flush, dim=0, out=_V149_FLUSH[1])
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def _v149_pair(candidate, incumbent, flush, repeats=13, rounds=3):
    ratios, cand, inc = [], [], []
    for r in range(rounds):
        if r & 1:
            b = _v149_time(incumbent, flush, repeats)
            a = _v149_time(candidate, flush, repeats)
        else:
            a = _v149_time(candidate, flush, repeats)
            b = _v149_time(incumbent, flush, repeats)
        ratios.append(max(a * 1000.0 - _V149_REPLAY_US, 0.05) /
                      max(b * 1000.0 - _V149_REPLAY_US, 0.05))
        cand.append(a * 1000.0)
        inc.append(b * 1000.0)
    ratios.sort(); cand.sort(); inc.sort()
    return ratios[len(ratios) // 2], cand[len(cand) // 2], inc[len(inc) // 2]


def _v149_validate(plan_state, weights):
    for m in (128, 256):
        for layer in ("qkv_gate", "o", "pooler", "unpooler"):
            plan = plan_state.get("plans", {}).get((m, layer))
            if plan is None or not plan[1].get("compressed", False):
                print("v149 rejected", m, layer, "missing exact C12 plan", flush=True)
                return False
            weight = weights[layer]
            gen = torch.Generator(device=weight.device).manual_seed(1490 + m + len(layer))
            x = torch.randn((m, weight.shape[1]), generator=gen,
                            dtype=torch.float32, device=weight.device).to(torch.bfloat16)
            x[-1].zero_(); x[-1, weight.shape[1] - 17] = 16
            out = _u32_run_m32w(x, layer, weight.shape[0], plan_state)
            if out is None or not bool(torch.isfinite(out).all()):
                print("v149 rejected", m, layer, "nonfinite or missing output", flush=True)
                return False
            ref = x[:8].double() @ weight.double().T
            diff = out[:8].double() - ref
            rel = float(diff.norm() / ref.norm())
            den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
            worst = float((diff.norm(dim=1) / den).max())
            if not (rel < 0.005 and worst < 0.01):
                print("v149 rejected", m, layer, "fp64", rel, worst, flush=True)
                return False
            print("v149 exact", m, layer, "fp64", rel, worst,
                  "regs", plan_state["ext"].info(plan[0]) // 10000,
                  "local", plan_state["ext"].info(plan[0]) % 10000,
                  "escapes", plan[1].get("escapes", -1),
                  "max", plan[1].get("max_esc", -1), flush=True)
    return True


_v105_prepare = prepare
_v105_kernel_fn = kernel_fn


def prepare(weights):
    state = _v105_prepare(weights)
    state["v149_enabled"] = set()
    state["v149_projected_gain"] = 0.0
    dev = weights["o"].device
    props = torch.cuda.get_device_properties(dev)
    if props.major != 9 or "H200" not in props.name:
        return state
    try:
        plan_state = _u32_prepare_m32w(
            {name: weights[name] for name in ("qkv_gate", "o", "pooler", "unpooler")},
            configs=_V149_CONFIGS,
        )
        state["v149"] = plan_state
        if not _v149_validate(plan_state, weights):
            return state
        flush = torch.ones(64 * 1024 * 1024 // 1024, 1024,
                           dtype=torch.float32, device=dev)
        route_times = {}
        qualified = set()
        load_ratios = []
        for m in (128, 256):
            inc_load = 0.0
            cand_load = 0.0
            for layer in ("qkv_gate", "o", "pooler", "unpooler"):
                weight = weights[layer]
                x = torch.randn((m, weight.shape[1]), device=dev).to(torch.bfloat16)
                cfg = {"layer": layer, "m": m, "n": weight.shape[0], "k": weight.shape[1]}
                ratio, cand_us, inc_us = _v149_pair(
                    lambda layer=layer, x=x, n=weight.shape[0]:
                        _u32_run_m32w(x, layer, n, plan_state),
                    lambda x=x, cfg=cfg: _v105_kernel_fn({"x": x, "config": cfg}, state),
                    flush,
                )
                keep = cand_us < inc_us and ratio <= _V149_ROUTE_RATIO
                if keep:
                    qualified.add((m, layer))
                route_times[m, layer] = (cand_us, inc_us, ratio, keep)
                inc_core = max(inc_us - _V149_REPLAY_US, 0.05)
                cand_core = max(cand_us - _V149_REPLAY_US, 0.05)
                inc_load += inc_core
                cand_load += cand_core if keep else inc_core
                print("v149 r128", m, layer, "candidate", cand_us,
                      "incumbent", inc_us, "core_ratio", ratio,
                      "qualified", keep, flush=True)
            rw = weights["router"]
            rx = torch.randn((m, rw.shape[1]), device=dev).to(torch.bfloat16)
            rcfg = {"layer": "router", "m": m, "n": rw.shape[0], "k": rw.shape[1]}
            router_us = _v149_time(
                lambda rx=rx, rcfg=rcfg: _v105_kernel_fn({"x": rx, "config": rcfg}, state),
                flush,
            )
            router_core = max(router_us * 1000.0 - _V149_REPLAY_US, 0.05)
            inc_load += router_core; cand_load += router_core
            load_ratios.append(inc_load / cand_load)
            print("v149 load", m, "incumbent", inc_load, "candidate", cand_load,
                  "ratio", inc_load / cand_load, flush=True)
        projected = (load_ratios[0] * load_ratios[1]) ** (1.0 / 6.0) - 1.0
        state["v149_projected_gain"] = projected
        if projected >= _V149_SCORE_GATE:
            state["v149_enabled"] = qualified
        print("v149 projected_score_gain", projected,
              "enabled", sorted(state["v149_enabled"]), flush=True)
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        print("v149 unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    key = (cfg["m"], cfg["layer"])
    if key in state.get("v149_enabled", ()):
        out = _u32_run_m32w(data["x"], cfg["layer"], cfg["n"], state["v149"])
        if out is not None:
            return out
    return _v105_kernel_fn(data, state)
'''

candidate = base + "\n" + prelude.lstrip() + "\n" + section + "\n\n" + integration.lstrip() + "\n"
compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print("baseline_prefix", candidate.encode()[:len(base_bytes)] == base_bytes)
print("bytes", len(candidate.encode()))
