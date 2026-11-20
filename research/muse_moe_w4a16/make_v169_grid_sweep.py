"""Build v169: lossless prepare-time WGS launch-grid tuner.

The production CUDA and all weight layouts remain byte-for-byte v167.  Only
M=32/128/256 may use a smaller resident grid, selected from paired cold-cache
CUDA-graph timings across deterministic routing seeds.  The 132-CTA launch is
the control and remains the fallback unless a candidate wins robustly.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v167_keell0renz_890809c.py"
DESTINATION = ROOT / "submission_v169_wgs_grid_tune.py"


APPEND = r'''


# ---------------------------------------------------------------------------
# v169: tune the resident WGS grid at prepare time.  The CUDA source, packed
# weights, arithmetic and kernel specialization are unchanged from v167.
# ---------------------------------------------------------------------------
_v169_parent_prepare = prepare
_v169_parent_kernel = kernel_fn
_V169_SHAPES = (32, 128, 256)
_V169_GRIDS = (96, 108, 120, 128, 132)
_V169_SEEDS = 5
_V169_REPS = 16


def _v169_graph(ext, state, x, ids, tw, out, grid):
    def fn():
        ext.okc_run(state["w"], state["scratch"], x, ids, tw, out,
                    state["lo"], state["cnt"], grid)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _v169_tune_grid(state, weights):
    dev = weights["w13"].device
    sm_grid = int(state["grid"])
    grids = tuple(g for g in _V169_GRIDS if g <= sm_grid)
    if sm_grid not in grids:
        grids += (sm_grid,)
    ext = _ext(tuple(_main_defines(state.get("sfx", ()))) + tuple(state.get("sfxm", state.get("sfx", ()))))
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    picks = {}
    for M in _V169_SHAPES:
        seed_medians = {g: [] for g in grids}
        paired = {g: [] for g in grids}
        max_diff = {g: 0.0 for g in grids}
        for seed_i in range(_V169_SEEDS):
            data = _gen_input(M, 9169 + 37 * seed_i + M, dev)
            x = data["hidden_states"].contiguous()
            ids = data["topk_ids"].to(torch.int32).contiguous()
            if ids.data_ptr() & 15:
                ids = ids.clone()
            tw = data["topk_weights"].to(torch.float32).contiguous()
            outs = {g: torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=dev) for g in grids}
            graphs = {g: _v169_graph(ext, state, x, ids, tw, outs[g], g) for g in grids}
            torch.cuda.synchronize()
            ref = outs[sm_grid].float()
            for g in grids:
                max_diff[g] = max(max_diff[g], float((outs[g].float() - ref).abs().max()))
            samples = {g: [] for g in grids}
            for rep in range(_V169_REPS):
                order = grids if rep % 2 == 0 else grids[::-1]
                for g in order:
                    flush.add_(1)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[g].replay()
                    e1.record()
                    torch.cuda.synchronize()
                    samples[g].append(e0.elapsed_time(e1) * 1000.0)
            ctl = samples[sm_grid]
            for g in grids:
                sg = sorted(samples[g])
                seed_medians[g].append(sg[len(sg) // 2])
                ratios = sorted(a / b for a, b in zip(ctl, samples[g]))
                paired[g].append(ratios[len(ratios) // 2])
        rows = []
        for g in grids:
            meds = sorted(seed_medians[g])
            rs = paired[g]
            rows.append((g, meds[len(meds) // 2], sorted(rs)[len(rs) // 2], sum(v > 1.0 for v in rs), max_diff[g]))
        # Require a meaningful paired median and wins on at least four of five
        # seeds.  This deliberately favors the known-safe full-SM control.
        eligible = [r for r in rows if r[2] >= 1.005 and r[3] >= 4 and r[4] <= 0.03125]
        best = max(eligible, key=lambda r: (r[2], -r[1])) if eligible else next(r for r in rows if r[0] == sm_grid)
        picks[M] = best[0]
        print("[v169-grid] M=%d %s pick=%d" %
              (M, " ".join("g%d %.1fus %.4fx %d/%d maxd %.5g" % (g, med, ratio, wins, _V169_SEEDS, diff)
                            for g, med, ratio, wins, diff in rows), best[0]), flush=True)
    state["v169_grid"] = picks


def prepare(weights):
    state = _v169_parent_prepare(weights)
    try:
        _v169_tune_grid(state, weights)
    except Exception as exc:
        state["v169_grid"] = {}
        print("[v169-grid] tuner failed; using full grid:", type(exc).__name__, str(exc)[-300:], flush=True)
    return state


def kernel_fn(data, state):
    M = int(data["hidden_states"].shape[0])
    grid = state.get("v169_grid", {}).get(M)
    if grid is None or M not in _V169_SHAPES or M > CHUNK_TOKENS:
        return _v169_parent_kernel(data, state)
    _bind_context(data["hidden_states"])
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    assert cfg["local_start"] == state["lo"] and cfg["local_count"] == state["cnt"]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    if ids.data_ptr() & 15:
        ids = ids.clone()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=x.device)
    sfx = state.get("sfx", ())
    ext = _ext(tuple(_main_defines(sfx)) + tuple(state.get("sfxm", sfx)))
    ext.okc_run(state["w"], state["scratch"], x, ids, tw, out,
                state["lo"], state["cnt"], int(grid))
    return out
'''


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    if "# okc: problem=muse_moe_w4a16" not in source.splitlines()[0]:
        raise RuntimeError("unexpected source header")
    if "# v169: tune the resident WGS grid" in source:
        raise RuntimeError("source already contains v169")
    DESTINATION.write_text(source.rstrip() + APPEND + "\n", encoding="utf-8", newline="\n")
    print(DESTINATION)


if __name__ == "__main__":
    main()
