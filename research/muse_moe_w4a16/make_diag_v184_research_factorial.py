"""Append a paired large-shape research-kernel factorial to protected v178."""

from pathlib import Path


BASE = Path(__file__).with_name("submission_v178_hybrid_m32_retry.py")
OUT = Path(__file__).with_name("submission_diag_v184_research_factorial.py")


APPEND = r'''


# Diagnostic only: exact v139 control versus the research WGS source with the
# two independently switchable mechanisms that changed its large-shape path.
# The official timed dispatch above remains byte-for-byte v178.
_RF_PARENT_PREPARE = prepare
_RF_SEEDS = 5
_RF_REPS = 16


def _rf_private_state(state, dev, research):
    clone = dict(state)
    clone["scratch"] = _v167_make_scratch(dev) if research else _make_scratch(dev)
    clone["buf_out"] = torch.empty((MMAX, 4096), dtype=torch.bfloat16, device=dev)
    return clone


def _rf_plain_research_state(weights, base):
    """Build the research source's exact non-SC6 KPERM3 WGS weight layout."""
    dev = weights["w13"].device
    lo, cnt = base["lo"], base["cnt"]
    I = weights["w13"].shape[1] // 2
    H = weights["w2"].shape[1]
    rb = torch.arange(I // 8, device=dev)
    g = torch.arange(8, device=dev)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(H, device=dev)
    assert _scales_prmt_exact(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt])
    state = dict(base)
    state["w"] = list(base["w"][:12]) + [
        torch.cat([_v167_wgs_copy(weights["w13"][e], weights["w13_sf"][e], _kperm_rows(rowsA), False)
                   for e in range(lo, lo + cnt)]),
        torch.cat([_v167_wgs_copy(weights["w2"][e], weights["w2_sf"][e], rowsB, False)
                   for e in range(lo, lo + cnt)]),
    ]
    state["sfx"] = ("-DOKC_PRMTSC=1",)
    state["use_mul"] = False
    return state


def _rf_factorial(parent, weights):
    import concurrent.futures
    import statistics
    import time

    t0 = time.time()
    dev = weights["w13"].device
    plain = _rf_plain_research_state(weights, parent["tiny"])
    sc6 = parent["m32"]
    control = _rf_private_state(parent["larger"], dev, False)
    states = {
        "control": control,
        "grid_plain": _rf_private_state(plain, dev, True),
        "mfin_plain": _rf_private_state(plain, dev, True),
        "grid_sc6": _rf_private_state(sc6, dev, True),
        "mfin_sc6": _rf_private_state(sc6, dev, True),
    }
    common = tuple(V167_MAIN_DEFINES) + ("-DOKC_PRMTSC=1",)
    defs = {
        "control": tuple(_main_defines(control["sfx"])) + tuple(control["sfx"]),
        "grid_plain": common + ("-DOKC_MFIN=0", "-DOKC_B4L=0", "@v167_main"),
        "mfin_plain": common + ("@v167_main",),
        "grid_sc6": common + ("-DOKC_SC6W=1", "-DOKC_MFIN=0", "-DOKC_B4L=0", "@v167_main"),
        "mfin_sc6": common + ("-DOKC_SC6W=1", "@v167_main"),
    }

    # v178 already built control and mfin_sc6. Compile the three missing cells
    # concurrently so the diagnostic remains under the prepare-time budget.
    missing = ("grid_plain", "mfin_plain", "grid_sc6")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        built = dict(zip(missing, pool.map(lambda n: _ext(defs[n], use_mul=False), missing)))
    exts = {
        "control": _ext(defs["control"], use_mul=control["use_mul"]),
        **built,
        "mfin_sc6": _ext(defs["mfin_sc6"], use_mul=False),
    }
    torch.cuda.synchronize()
    print(f"[rf] five cells resolved in {time.time() - t0:.1f}s", flush=True)

    names = ("control", "grid_plain", "mfin_plain", "grid_sc6", "mfin_sc6")
    candidates = names[1:]
    shape_medians = {M: {name: [] for name in names} for M in (128, 256)}
    shape_ratios = {M: {name: [] for name in candidates} for M in (128, 256)}
    rels = {name: [] for name in candidates}
    rows = {name: [] for name in candidates}
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)

    for si in range(_RF_SEEDS):
        for M in (128, 256):
            data = _gen_input(M, 184003 + si * 1009 + M * 17, dev)
            x = data["hidden_states"].contiguous()
            ids = data["topk_ids"].to(torch.int32).contiguous()
            tw = data["topk_weights"].float().contiguous()
            outputs = {name: torch.empty((M, 4096), dtype=torch.bfloat16, device=dev) for name in names}
            graphs = {}
            for name in names:
                st = states[name]

                def fn(name=name, st=st):
                    exts[name].okc_run(st["w"], st["scratch"], x, ids, tw, outputs[name],
                                      st["lo"], st["cnt"], st["grid"])

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

            ref = outputs["control"].float()
            den = ref.norm().clamp_min(1e-30)
            row_den = ref.norm(dim=1).clamp_min(ref.norm(dim=1).mean() * 0.1)
            for name in candidates:
                diff = outputs[name].float() - ref
                rels[name].append(float(diff.norm() / den))
                rows[name].append(float((diff.norm(dim=1) / row_den).max()))

            samples = {name: [] for name in names}
            for rep in range(_RF_REPS):
                shift = (rep + si + (M >> 7)) % len(names)
                order = names[shift:] + names[:shift]
                if rep & 1:
                    order = order[::-1]
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
                shape_medians[M][name].append(statistics.median(samples[name]))
            for name in candidates:
                shape_ratios[M][name].append(statistics.median(
                    a / b for a, b in zip(samples[name], samples["control"])
                ))

    for M in (128, 256):
        print(f"[rf] M{M} control={statistics.median(shape_medians[M]['control']):.1f}us", flush=True)
        for name in candidates:
            ratios = shape_ratios[M][name]
            print(
                f"[rf] M{M} {name}={statistics.median(shape_medians[M][name]):.1f}us "
                f"ratio={statistics.median(ratios):.5f} wins={sum(r < 1.0 for r in ratios)}/{len(ratios)} "
                f"per-seed={' '.join(f'{r:.5f}' for r in ratios)}",
                flush=True,
            )

    ranking = []
    for name in candidates:
        verify = []
        for si in range(_RF_SEEDS):
            csum = 65.7 + shape_medians[128]["control"][si] + shape_medians[256]["control"][si]
            vsum = 65.7 + shape_medians[128][name][si] + shape_medians[256][name][si]
            verify.append(vsum / csum)
        vr = statistics.median(verify)
        projected = statistics.median(r ** 0.5 for r in verify)
        win_count = sum(r < 1.0 for r in verify)
        promote = vr <= 0.985 and win_count >= 4 and max(rels[name]) <= 0.015 and max(rows[name]) <= 0.075
        ranking.append((vr, name))
        print(
            f"[rf] summary {name}: verify={vr:.5f} projected-score={projected:.5f} "
            f"wins={win_count}/{len(verify)} rel-max={max(rels[name]):.5f} row-max={max(rows[name]):.5f} "
            f"PROMOTE={promote} per-seed={' '.join(f'{r:.5f}' for r in verify)}",
            flush=True,
        )
    ranking.sort()
    print(f"[rf] BEST={ranking[0][1]} verify={ranking[0][0]:.5f} total={time.time() - t0:.1f}s", flush=True)
    assert max(max(v) for v in rels.values()) <= 0.015
    assert max(max(v) for v in rows.values()) <= 0.075


def prepare(weights):
    state = _RF_PARENT_PREPARE(weights)
    try:
        _rf_factorial(state, weights)
    except Exception as ex:
        print(f"[rf] failed: {str(ex)[-5000:]}", flush=True)
    return state
'''


def main():
    raw = BASE.read_bytes()
    source = raw.decode("utf-8")
    assert source.startswith("# okc: problem=muse_moe_w4a16")
    assert "_RF_PARENT_PREPARE" not in source
    OUT.write_bytes(raw.rstrip(b"\r\n") + APPEND.encode("utf-8"))
    print(OUT)


if __name__ == "__main__":
    main()
