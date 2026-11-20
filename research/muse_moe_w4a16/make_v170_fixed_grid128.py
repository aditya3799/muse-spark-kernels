"""Build v170: fixed 128-CTA WGS diagnostic with clean scratch state."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v167_keell0renz_890809c.py"
DESTINATION = ROOT / "submission_v170_wgs_grid128.py"


APPEND = r'''


# v170: fixed 128-CTA WGS diagnostic.  Small/MID shapes retain the production
# 132-CTA dispatch.  Reallocate scratch after prepare's 132-CTA diagnostics so
# no generation/completion state crosses launch geometries.
_v170_parent_prepare = prepare
_v170_parent_kernel = kernel_fn
_V170_GRID = 128


def prepare(weights):
    state = _v170_parent_prepare(weights)
    state["scratch"] = _make_scratch(weights["w13"].device)
    print("[v170-grid] clean scratch; WGS grid", _V170_GRID, flush=True)
    return state


def kernel_fn(data, state):
    M = int(data["hidden_states"].shape[0])
    if M not in (32, 128, 256) or M > CHUNK_TOKENS:
        return _v170_parent_kernel(data, state)
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
                state["lo"], state["cnt"], _V170_GRID)
    return out
'''


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    DESTINATION.write_text(text.rstrip() + APPEND + "\n", encoding="utf-8", newline="\n")
    print(DESTINATION)


if __name__ == "__main__":
    main()
