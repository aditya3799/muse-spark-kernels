"""Append paired cold-L2 M128/M256 control-vs-large-B4 timing to v183."""

from pathlib import Path


BASE = Path(__file__).with_name("submission_v183_hybrid_large_b4.py")
OUT = Path(__file__).with_name("submission_diag_v183_large_b4_pair.py")


APPEND = r'''


# Diagnostic-only large-shape comparison. Official timed M33..M256 uses S052 BIGB4.
_LB4_PARENT_PREPARE = prepare
_LB4_SEEDS = 5
_LB4_REPS = 20


def _lb4_pair(state, weights):
    import statistics
    import time

    t0 = time.time()
    control = state["larger"]
    candidate = state["large_b4"]
    control_defs = tuple(_main_defines(control["sfx"])) + tuple(control["sfx"])
    candidate_defs = (tuple(S052_MAIN_DEFINES) + tuple(candidate["sfx"]) +
                      ("-DOKC_B4FIN=1", "-DOKC_BIGB4=1", "@s052_large_b4"))
    exts = {
        "control": _ext(control_defs, use_mul=control["use_mul"]),
        "bigb4": _ext(candidate_defs, use_mul=False),
    }
    print(f"[lb4] resolved prebuilt control,bigb4 at {time.time() - t0:.1f}s", flush=True)

    dev = weights["w13"].device
    states = {"control": control, "bigb4": candidate}
    names = ("control", "bigb4")
    shape_medians = {128: {name: [] for name in names}, 256: {name: [] for name in names}}
    shape_ratios = {128: [], 256: []}
    rels = []
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)

    for si in range(_LB4_SEEDS):
        for M in (128, 256):
            data = _gen_input(M, 183003 + si * 1009 + M * 17, dev)
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
            den = outputs["control"].float().norm().clamp_min(1e-30)
            rels.append(float((outputs["bigb4"].float() - outputs["control"].float()).norm() / den))
            samples = {name: [] for name in names}
            for rep in range(_LB4_REPS):
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
                shape_medians[M][name].append(statistics.median(samples[name]))
            shape_ratios[M].append(statistics.median(
                b / a for a, b in zip(samples["control"], samples["bigb4"])
            ))

    for M in (128, 256):
        cm = statistics.median(shape_medians[M]["control"])
        bm = statistics.median(shape_medians[M]["bigb4"])
        ratios = shape_ratios[M]
        print(
            f"[lb4] M{M} control={cm:.1f}us bigb4={bm:.1f}us ratio={statistics.median(ratios):.5f} "
            f"wins={sum(r < 1.0 for r in ratios)}/{len(ratios)} "
            f"per-seed={' '.join(f'{r:.5f}' for r in ratios)}",
            flush=True,
        )
    verify_ratios = []
    for si in range(_LB4_SEEDS):
        csum = 65.7 + shape_medians[128]["control"][si] + shape_medians[256]["control"][si]
        bsum = 65.7 + shape_medians[128]["bigb4"][si] + shape_medians[256]["bigb4"][si]
        verify_ratios.append(bsum / csum)
    projected = statistics.median(r ** 0.5 for r in verify_ratios)
    print(
        f"[lb4] verify-ratio={statistics.median(verify_ratios):.5f} projected-score-ratio={projected:.5f} "
        f"wins={sum(r < 1.0 for r in verify_ratios)}/{len(verify_ratios)} "
        f"per-seed={' '.join(f'{r:.5f}' for r in verify_ratios)} rel-max={max(rels):.3e}",
        flush=True,
    )
    assert max(rels) < 1e-2, max(rels)


def prepare(weights):
    state = _LB4_PARENT_PREPARE(weights)
    try:
        _lb4_pair(state, weights)
    except Exception as ex:
        print(f"[lb4] failed: {str(ex)[-3000:]}", flush=True)
    return state
'''


def main():
    raw = BASE.read_bytes()
    src = raw.decode("utf-8")
    assert src.startswith("# okc: problem=muse_moe_w4a16")
    assert "submission_diag_v183_large_b4_pair" not in src
    OUT.write_bytes(raw.rstrip(b"\r\n") + APPEND.encode("utf-8"))
    print(OUT)


if __name__ == "__main__":
    main()
