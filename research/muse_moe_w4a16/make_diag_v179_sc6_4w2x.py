from pathlib import Path


BASE = Path(__file__).with_name("submission_v167_keell0renz_890809c.py")
OUT = Path(__file__).with_name("submission_diag_v179_sc6_4w2x.py")


APPEND = r"""


# ---------------------------------------------------------------------------
# Diagnostic-only SC6 split-ring experiment.  The production v167 kernel_fn is
# unchanged.  The twin packs each SC6 weight slot to its true 35,840-byte size,
# then spends the recovered 4 KiB on a fourth weight stage while reducing the
# activation ring from three stages to two.  Total dynamic shared memory is
# identical to production:
#   prod: 3 * 36,864 + 3 * 32,768 = 208,896 B
#   twin: 4 * 35,840 + 2 * 32,768 = 208,896 B
_W42_OLD_CONSTANTS = '''#define OKC_WGSNX 3                             // x ring slots
constexpr int WGS_NS = 3;                       // weight ring slots (36 KB; freed as soon as the codes are in registers)
constexpr int WGS_NX = OKC_WGSNX;               // x ring slots (freed when the stage's last wgmma retired)
constexpr int WGS_NT = FT_T / 8;                // max n-tiles of a job
constexpr int WGS_PIECE = OKC_SC6W ? 8960 : 9216;  // one warpgroup's weights in a stage (SC6: 8 KB codes + 768 B scales)
constexpr int WGS_STG = OKC_SC6W ? 35840 : 36864;  // bytes of a stage in global memory (the smem slot keeps UNIT_BYTES)
constexpr int WGS_XS = WGS_NT * 4096;           // an x slot: up to FT_T rows x 256 K
constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;  // barrier indices'''
_W42_NEW_CONSTANTS = '''#ifndef OKC_WGSNS
#define OKC_WGSNS 3                             // weight ring slots
#endif
#ifndef OKC_WGSNX
#define OKC_WGSNX 3                             // x ring slots
#endif
constexpr int WGS_NS = OKC_WGSNS;               // weight ring slots; released once codes/scales are in registers
constexpr int WGS_NX = OKC_WGSNX;               // x ring slots; released after the stage's last wgmma retires
static_assert(WGS_NS >= 2 && WGS_NX >= 2, "WGS rings must remain pipelined");
constexpr int WGS_NT = FT_T / 8;                // max n-tiles of a job
constexpr int WGS_PIECE = OKC_SC6W ? 8960 : 9216;  // one warpgroup's weights in a stage (SC6: 8 KB codes + 768 B scales)
constexpr int WGS_STG = OKC_SC6W ? 35840 : 36864;  // exact global and shared bytes in one packed weight stage
constexpr int WGS_XS = WGS_NT * 4096;           // an x slot: up to FT_T rows x 256 K
constexpr int BW_F = 0, BW_E = WGS_NS, BX_F = 2 * WGS_NS, BX_E = 2 * WGS_NS + WGS_NX;  // barrier indices'''

assert CUDA_SRC.count(_W42_OLD_CONSTANTS) == 1
_w42 = CUDA_SRC.replace(_W42_OLD_CONSTANTS, _W42_NEW_CONSTANTS, 1)

_W42_REPL = (
    (
        '::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_ARG : "memory");',
        '::"r"(smem_u32(ring + ws * WGS_STG)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_ARG : "memory");',
    ),
    ('unsigned char* xring = ring + WGS_NS * UNIT_BYTES;',
     'unsigned char* xring = ring + WGS_NS * WGS_STG;'),
    ('const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);',
     'const uint32_t xring_s = smem_u32(ring + WGS_NS * WGS_STG);'),
    ('const unsigned char* pc = pc0 + ws * UNIT_BYTES;',
     'const unsigned char* pc = pc0 + ws * WGS_STG;'),
    ('return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;',
     'return base + (size_t)okc::WGS_NS * okc::WGS_STG + (size_t)okc::WGS_NX * okc::WGS_XS;'),
)
for _old, _new in _W42_REPL:
    assert _w42.count(_old) == 1, (_old, _w42.count(_old))
    _w42 = _w42.replace(_old, _new, 1)
ALT_SRCS["sc6_4w2x"] = _w42

W42_SHAPES = (32, 128, 256)
W42_SEEDS = 5
W42_REPS = 16


def _w42_defs(state):
    base = tuple(_main_defines(state["sfx"])) + tuple(state.get("sfxm", state["sfx"]))
    return {
        "prod": base,
        "w4x2": base + ("-DOKC_WGSNS=4", "-DOKC_WGSNX=2", "@sc6_4w2x"),
    }


def _w42_time(state, weights, exts, bs):
    import statistics

    dev = weights["w13"].device
    names = ("prod", "w4x2")
    per_seed = {n: [] for n in names}
    ratios = []
    rels = []
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    for si in range(W42_SEEDS):
        d = _gen_input(bs, 179003 + bs * 31 + si * 1009, dev)
        x = d["hidden_states"].contiguous()
        ids = d["topk_ids"].to(torch.int32).contiguous()
        tw = d["topk_weights"].float().contiguous()
        outs = {n: torch.empty((bs, HIDDEN), dtype=torch.bfloat16, device=dev) for n in names}
        scratches = {n: _make_scratch(dev) for n in names}
        graphs = {}
        for n in names:
            def fn(n=n):
                exts[n].okc_run(state["w"], scratches[n], x, ids, tw, outs[n], state["lo"], state["cnt"], state["grid"])
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
        den = outs["prod"].float().norm().clamp_min(1e-30)
        rels.append(float((outs["w4x2"].float() - outs["prod"].float()).norm() / den))
        samples = {n: [] for n in names}
        for rep in range(W42_REPS):
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
            per_seed[n].append(statistics.median(samples[n]))
        ratios.append(statistics.median(a / b for a, b in zip(samples["prod"], samples["w4x2"])))
    pmed = statistics.median(per_seed["prod"])
    cmed = statistics.median(per_seed["w4x2"])
    rmed = statistics.median(ratios)
    print(
        f"[w42] bs={bs:3d} prod={pmed:.1f}us w4x2={cmed:.1f}us "
        f"prod/w4x2={rmed:.4f} per-seed={' '.join(f'{v:.4f}' for v in ratios)} "
        f"rel-max={max(rels):.3e}",
        flush=True,
    )


def _w42_diag(state, weights):
    import concurrent.futures
    import time

    t0 = time.time()
    defs = _w42_defs(state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futs = {n: pool.submit(_ext, d) for n, d in defs.items()}
        exts = {n: f.result() for n, f in futs.items()}
    print(
        "[w42] builds " + " ".join(f"{n}={exts[n].okc_attrs()}" for n in defs)
        + " ring-bytes(prod=208896,w4x2=208896)"
        + f" at {time.time() - t0:.1f}s",
        flush=True,
    )
    for bs in W42_SHAPES:
        _w42_time(state, weights, exts, bs)
    print(f"[w42] total {time.time() - t0:.1f}s", flush=True)


_w42_parent_prepare = prepare


def prepare(weights):
    state = _w42_parent_prepare(weights)
    try:
        _w42_diag(state, weights)
    except Exception as ex:
        print(f"[w42] failed: {str(ex)[-3000:]}", flush=True)
    return state
"""


def main():
    raw = BASE.read_bytes()
    src = raw.decode("utf-8")
    assert src.startswith("# okc: problem=muse_moe_w4a16")
    assert "submission_diag_v179_sc6_4w2x" not in src
    OUT.write_bytes(raw.rstrip(b"\r\n") + APPEND.encode("utf-8") + b"\n")
    print(OUT)


if __name__ == "__main__":
    main()
