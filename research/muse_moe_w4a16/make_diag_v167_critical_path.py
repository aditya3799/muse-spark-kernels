from pathlib import Path


BASE = Path(__file__).with_name("submission_v167_keell0renz_890809c.py")
OUT = Path(__file__).with_name("submission_diag_v167_critical_path.py")


APPEND = r'''


# ---------------------------------------------------------------------------
# Diagnostic-only wrapper: exact v167 production build plus a valid-output
# global-barrier twin and an instrumented production twin.  This is deliberately
# appended after every production wrapper, so kernel_fn and its timed dispatch
# are byte-for-byte unchanged.  It runs only from prepare() in this diagnostic
# candidate and uses private scratch for every variant.
_CRIT_DBG_ANCHOR = "#define OKC_DEBUG 0"
assert CUDA_SRC.count(_CRIT_DBG_ANCHOR) == 1
ALT_SRCS["crit_dbg"] = CUDA_SRC.replace(
    _CRIT_DBG_ANCHOR,
    "#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif",
    1,
)

CRIT_SHAPES = (32, 128, 256)
CRIT_SEEDS = 3
CRIT_REPS = 12


def _crit_defs(state):
    base = tuple(_main_defines(state["sfx"]))
    prod = base + tuple(state.get("sfxm", state["sfx"]))
    sync = tuple(d for d in base if not d.startswith(("-DOKC_MFIN", "-DOKC_B4L", "-DOKC_MFS")))
    sync += ("-DOKC_MFIN=0", "-DOKC_B4L=0") + tuple(state.get("sfxm", state["sfx"]))
    dbg = prod + ("-DOKC_DEBUG=1", "@crit_dbg")
    return {"prod": prod, "sync": sync, "debug": dbg}


def _crit_timeline(ext, state, weights, bs, seed):
    dev = weights["w13"].device
    d = _gen_input(bs, seed, dev)
    x = d["hidden_states"].contiguous()
    ids = d["topk_ids"].to(torch.int32).contiguous()
    tw = d["topk_weights"].float().contiguous()
    out = torch.empty((bs, HIDDEN), dtype=torch.bfloat16, device=dev)
    G = state["grid"]
    dbg = torch.zeros(G * 8 + G * 8 * 4, dtype=torch.int64, device=dev)
    scratch = list(_make_scratch(dev)) + [dbg]
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    # Establish the launch generation, then inspect one cold-L2 launch.
    for _ in range(3):
        ext.okc_run(state["w"], scratch, x, ids, tw, out, state["lo"], state["cnt"], G)
    torch.cuda.synchronize()
    flush.add_(1)
    dbg.zero_()
    torch.cuda.synchronize()
    ext.okc_run(state["w"], scratch, x, ids, tw, out, state["lo"], state["cnt"], G)
    torch.cuda.synchronize()
    tt = dbg[:G * 8].view(G, 8).cpu().double()

    def col(k):
        v = tt[:, k]
        return v[v > 0]

    t0 = col(0).min()
    vals = {}
    for k, name in ((1, "routed"), (6, "gather"), (2, "A"), (3, "B"), (4, "ready"), (5, "end")):
        v = col(k)
        if v.numel():
            vals[name] = ((v.min() - t0) / 1000.0, (v.max() - t0) / 1000.0)
    fmt = " ".join(f"{name}={lo:.1f}-{hi:.1f}" for name, (lo, hi) in vals.items())
    print(f"[crit] timeline bs={bs:3d} seed={seed}: {fmt}", flush=True)
    return vals


def _crit_time(state, weights, exts, bs):
    import statistics

    dev = weights["w13"].device
    names = ("prod", "sync")
    per_seed = {n: [] for n in names}
    ratios = []
    max_rel = 0.0
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    for si in range(CRIT_SEEDS):
        d = _gen_input(bs, 91001 + bs * 17 + si * 101, dev)
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
        max_rel = max(max_rel, float((outs["sync"].float() - outs["prod"].float()).norm() / den))
        samples = {n: [] for n in names}
        for rep in range(CRIT_REPS):
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
        ratios.append(statistics.median(a / b for a, b in zip(samples["prod"], samples["sync"])))
    pmed = statistics.median(per_seed["prod"])
    smed = statistics.median(per_seed["sync"])
    rmed = statistics.median(ratios)
    print(
        f"[crit] paired bs={bs:3d}: prod={pmed:.1f}us sync={smed:.1f}us "
        f"prod/sync={rmed:.4f} per-seed={' '.join(f'{v:.4f}' for v in ratios)} rel={max_rel:.3e}",
        flush=True,
    )


def _critical_diag(state, weights):
    import concurrent.futures
    import time

    t0 = time.time()
    defs = _crit_defs(state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futs = {n: pool.submit(_ext, d) for n, d in defs.items()}
        exts = {n: f.result() for n, f in futs.items()}
    print(
        "[crit] builds "
        + " ".join(f"{n}={exts[n].okc_attrs()}" for n in defs)
        + f" at {time.time() - t0:.1f}s",
        flush=True,
    )
    for bs in CRIT_SHAPES:
        _crit_time(state, weights, exts, bs)
        for si in range(CRIT_SEEDS):
            _crit_timeline(exts["debug"], state, weights, bs, 93001 + bs * 19 + si * 103)
    print(f"[crit] total {time.time() - t0:.1f}s", flush=True)


_critical_parent_prepare = prepare


def prepare(weights):
    state = _critical_parent_prepare(weights)
    try:
        _critical_diag(state, weights)
    except Exception as ex:
        print(f"[crit] failed: {str(ex)[-2000:]}", flush=True)
    return state
'''


def main():
    raw = BASE.read_bytes()
    src = raw.decode("utf-8")
    assert src.startswith("# okc: problem=muse_moe_w4a16")
    assert "submission_diag_v167_critical_path" not in src
    # Keep every production byte (including LF line endings) intact.  The
    # diagnostic is an appendix, not a rewrite of the candidate.
    OUT.write_bytes(raw.rstrip(b"\r\n") + APPEND.encode("utf-8") + b"\n")
    print(OUT)


if __name__ == "__main__":
    main()
