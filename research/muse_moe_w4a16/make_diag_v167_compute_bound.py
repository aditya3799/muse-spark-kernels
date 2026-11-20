from pathlib import Path


BASE = Path(__file__).with_name("submission_v167_keell0renz_890809c.py")
OUT = Path(__file__).with_name("submission_diag_v167_compute_bound.py")


APPEND = r"""


# ---------------------------------------------------------------------------
# Diagnostic-only compute bounds for the exact v167 MAIN/SC6 path.  The timed
# production kernel_fn above is unchanged.  Both invalid-output twins run only
# during prepare() and own private scratch.
_CB_WGS0 = "#define OKC_WGSX 0  // dev: 1 = WGS without the wgmma, 2 = WGS phases skipped"
assert CUDA_SRC.count(_CB_WGS0) == 1
_cb_base = CUDA_SRC.replace(
    _CB_WGS0,
    _CB_WGS0 + "\n#ifndef OKC_WGSRAWDQ\n#define OKC_WGSRAWDQ 0\n#endif",
    1,
)

_CB_DQ = '''          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
          // scales stored [j][h] (rows g, g+8 of step j side by side): one PRMT + one IMAD build both rows' scales in
          // one bf16x2, and the multiplies broadcast its halves (HMUL2 .H0_H0 / .H1_H1): 2 instructions per k16 fewer
#if OKC_SC6W
          const uint32_t sgp = sgs[js];
#else
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
#endif
          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);'''
assert _cb_base.count(_CB_DQ) == 1
_CB_RAW = '''#if OKC_WGSRAWDQ
          af[ci & 1][jj][0] = q;
          af[ci & 1][jj][1] = __funnelshift_r(q, q, 8);
          af[ci & 1][jj][2] = __funnelshift_r(q, q, 16);
          af[ci & 1][jj][3] = __funnelshift_r(q, q, 24);
#else
''' + _CB_DQ + '''
#endif'''
ALT_SRCS["cb_raw"] = _cb_base.replace(_CB_DQ, _CB_RAW, 1)

_CB_BEG = "template <int NTT> struct WGSMMA;"
_CB_END = "// two producer warps: the weights warp streams every stage's"
assert _cb_base.count(_CB_BEG) == 1 and _cb_base.count(_CB_END) == 1
_cb_i = _cb_base.index(_CB_BEG)
_cb_j = _cb_base.index(_CB_END, _cb_i)
_CB_STUB = r'''template <int NTT> struct WGSMMA {
  static __device__ __forceinline__ void consume(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
    uint32_t sink;
    asm volatile("{ .reg .pred p;\n"
                 "  lop3.b32 %0, %1, %2, %3, 0x96;\n"
                 "  xor.b32 %0, %0, %4;\n"
                 "  and.b32 %0, %0, %5;\n"
                 "  setp.ne.u32 p, %0, 0;\n"
                 "  @p trap;\n}\n"
                 : "=r"(sink) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(0u));
  }
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t) {
#pragma unroll
    for (int nt = 0; nt < WGS_NT; ++nt) {
#pragma unroll
      for (int q = 0; q < 4; ++q) d[nt][q] = 0.0f;
    }
    consume(a0, a1, a2, a3);
  }
  static __device__ __forceinline__ void mma(float (&)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t) {
    consume(a0, a1, a2, a3);
  }
};
'''
ALT_SRCS["cb_nowg"] = _cb_base[:_cb_i] + _CB_STUB + _cb_base[_cb_j:]

CB_SHAPES = (32, 128, 256)
CB_SEEDS = 5
CB_REPS = 16


def _cb_defs(state):
    base = tuple(_main_defines(state["sfx"])) + tuple(state.get("sfxm", state["sfx"]))
    return {
        "prod": base,
        "rawdq": base + ("-DOKC_WGSRAWDQ=1", "@cb_raw"),
        "nowg": base + ("@cb_nowg",),
    }


def _cb_time(state, weights, exts, bs):
    import statistics

    dev = weights["w13"].device
    names = tuple(exts)
    medians = {n: [] for n in names}
    ratios = {n: [] for n in names if n != "prod"}
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    for si in range(CB_SEEDS):
        d = _gen_input(bs, 101003 + bs * 29 + si * 1009, dev)
        x = d["hidden_states"].contiguous()
        ids = d["topk_ids"].to(torch.int32).contiguous()
        tw = d["topk_weights"].float().contiguous()
        outputs = {n: torch.empty((bs, HIDDEN), dtype=torch.bfloat16, device=dev) for n in names}
        scratches = {n: _make_scratch(dev) for n in names}
        graphs = {}
        for n in names:
            def fn(n=n):
                exts[n].okc_run(state["w"], scratches[n], x, ids, tw, outputs[n], state["lo"], state["cnt"], state["grid"])
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graphs[n] = graph
        torch.cuda.synchronize()
        samples = {n: [] for n in names}
        for rep in range(CB_REPS):
            order = names if rep % 2 == 0 else names[::-1]
            for n in order:
                flush.add_(1)
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[n].replay()
                e1.record()
                torch.cuda.synchronize()
                samples[n].append(e0.elapsed_time(e1) * 1000.0)
        for n in names:
            medians[n].append(statistics.median(samples[n]))
        for n in ratios:
            ratios[n].append(statistics.median(a / b for a, b in zip(samples["prod"], samples[n])))
    desc = " ".join(f"{n}={statistics.median(medians[n]):.1f}us" for n in names)
    bounds = " ".join(
        f"prod/{n}={statistics.median(ratios[n]):.4f}[{' '.join(f'{v:.4f}' for v in ratios[n])}]"
        for n in ratios
    )
    print(f"[cb] bs={bs:3d} {desc} {bounds}", flush=True)


def _compute_bound_diag(state, weights):
    import concurrent.futures
    import time

    t0 = time.time()
    defs = _cb_defs(state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futs = {n: pool.submit(_ext, d) for n, d in defs.items()}
        exts = {n: f.result() for n, f in futs.items()}
    print("[cb] builds " + " ".join(f"{n}={exts[n].okc_attrs()}" for n in defs), flush=True)
    for bs in CB_SHAPES:
        _cb_time(state, weights, exts, bs)
    print(f"[cb] total {time.time() - t0:.1f}s", flush=True)


_cb_parent_prepare = prepare


def prepare(weights):
    state = _cb_parent_prepare(weights)
    try:
        _compute_bound_diag(state, weights)
    except Exception as ex:
        print(f"[cb] failed: {str(ex)[-2000:]}", flush=True)
    return state
"""


def main():
    raw = BASE.read_bytes()
    src = raw.decode("utf-8")
    assert src.startswith("# okc: problem=muse_moe_w4a16")
    assert "submission_diag_v167_compute_bound" not in src
    OUT.write_bytes(raw.rstrip(b"\r\n") + APPEND.encode("utf-8") + b"\n")
    print(OUT)


if __name__ == "__main__":
    main()
