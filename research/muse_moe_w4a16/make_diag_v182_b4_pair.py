"""Append a seven-seed M32 control/B4FIN paired graph diagnostic to v182."""

from pathlib import Path


BASE = Path(__file__).with_name("submission_v182_hybrid_m32_decoder_b4.py")
OUT = Path(__file__).with_name("submission_diag_v182_b4_pair.py")


APPEND = r'''


# Diagnostic-only M32 comparison. The official timed path above is v182 B4FIN.
_B4_PARENT_PREPARE = prepare
_B4_SEEDS = 7
_B4_REPS = 24


def _b4_pair(state, weights):
    import statistics
    import time

    t0 = time.time()
    st = state["m32"]
    sfx = tuple(st["sfx"])
    defs = {
        "control": tuple(V167_MAIN_DEFINES) + sfx + ("@v167_main",),
        "b4": tuple(V167_MAIN_DEFINES) + sfx + ("-DOKC_B4FIN=1", "@v182_decoder_b4"),
    }
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futs = {name: pool.submit(_ext, d, False) for name, d in defs.items()}
        exts = {name: fut.result() for name, fut in futs.items()}
    # load_inline exposes only the functions requested by _ext (okc_run and
    # okc_set_pdl).  Kernel attributes were already established by the local
    # SASS scout, so keep the remote diagnostic on the selector's public API.
    print(f"[b4] compiled control,b4 at {time.time() - t0:.1f}s", flush=True)

    dev = weights["w13"].device
    names = ("control", "b4")
    medians = {name: [] for name in names}
    ratios = []
    rels = []
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    for si in range(_B4_SEEDS):
        data = _gen_input(32, 182003 + si * 1009, dev)
        x = data["hidden_states"].contiguous()
        ids = data["topk_ids"].to(torch.int32).contiguous()
        tw = data["topk_weights"].float().contiguous()
        outputs = {name: torch.empty((32, 4096), dtype=torch.bfloat16, device=dev) for name in names}
        scratches = {name: _v167_make_scratch(dev) for name in names}
        graphs = {}
        for name in names:
            def fn(name=name):
                exts[name].okc_run(st["w"], scratches[name], x, ids, tw, outputs[name], st["lo"], st["cnt"], st["grid"])
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graphs[name] = graph
        torch.cuda.synchronize()
        den = outputs["control"].float().norm().clamp_min(1e-30)
        rels.append(float((outputs["b4"].float() - outputs["control"].float()).norm() / den))
        samples = {name: [] for name in names}
        for rep in range(_B4_REPS):
            order = names if rep % 2 == 0 else names[::-1]
            for name in order:
                flush.add_(1)
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[name].replay()
                e1.record()
                torch.cuda.synchronize()
                samples[name].append(e0.elapsed_time(e1) * 1000.0)
        for name in names:
            medians[name].append(statistics.median(samples[name]))
        ratios.append(statistics.median(b / a for a, b in zip(samples["control"], samples["b4"])))
    cm = statistics.median(medians["control"])
    bm = statistics.median(medians["b4"])
    rm = statistics.median(ratios)
    print(
        f"[b4] M32 control={cm:.1f}us b4={bm:.1f}us b4/control={rm:.5f} "
        f"wins={sum(r < 1.0 for r in ratios)}/{len(ratios)} "
        f"per-seed={' '.join(f'{r:.5f}' for r in ratios)} rel-max={max(rels):.3e}",
        flush=True,
    )
    assert max(rels) < 1e-3, max(rels)


def prepare(weights):
    state = _B4_PARENT_PREPARE(weights)
    try:
        _b4_pair(state, weights)
    except Exception as ex:
        print(f"[b4] failed: {str(ex)[-3000:]}", flush=True)
    return state
'''


def main():
    raw = BASE.read_bytes()
    src = raw.decode("utf-8")
    assert src.startswith("# okc: problem=muse_moe_w4a16")
    assert "submission_diag_v182_b4_pair" not in src
    OUT.write_bytes(raw.rstrip(b"\r\n") + APPEND.encode("utf-8"))
    print(OUT)


if __name__ == "__main__":
    main()
