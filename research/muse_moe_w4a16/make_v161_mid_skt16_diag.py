"""Build the v161 diagnostic from the frozen v139 production candidate."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v139_wgs_kperm3.py"
TARGET = ROOT / "submission_v161_mid_skt16_diag.py"
EXPECTED = "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09"


CONFIG_OLD = '''DIAG = False               # prepare() prints an H200 timing/diagnostic table (untimed; for development only)
DIAG_VARIANTS = {}         # name -> defines, timed in diag mode
'''

CONFIG_NEW = '''MID_SKT16_DEFINES = (
    "-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_SPLIT=1", "-DOKC_SKT=16",
    "-DOKC_CPF=0", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4",
    "-DOKC_UWARP=0", "-DOKC_WS=0", "-DOKC_FINPF=1",
    "-DOKC_SKSC2=1", "-DOKC_FTSC2=1",
)
DIAG = True                # diagnostic-only: official kernel_fn remains the frozen v139 dispatch
DIAG_VARIANTS = {"skt16": MID_SKT16_DEFINES}
'''


DIAG_FUNCTION = r'''def _diag(state, weights):
    """Twenty-one-seed, cold-L2, graph-replay comparison of MID8 and SKT16."""
    import concurrent.futures
    import time

    dev = weights["w13"].device
    t0 = time.time()
    bs = 16
    seeds = tuple(1234 + 1009 * i for i in range(21))
    sfx = tuple(state.get("sfx", ()))
    variants = {"mid8": tuple(MID_DEFINES) + sfx}
    variants.update({name: tuple(defs) + sfx for name, defs in DIAG_VARIANTS.items()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(variants)) as ex:
        list(ex.map(_ext, [defs for defs in variants.values() if tuple(defs) not in _EXTS]))
    print(f"[v161] compiled {len(variants)} M16 variants in {time.time() - t0:.0f}s", flush=True)

    exts = {name: _ext(defs) for name, defs in variants.items()}
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    ratios = []
    saved_ratios = []
    unsplit_ratios = []
    wins = 0

    def capture(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        return graph

    for si, seed0 in enumerate(seeds):
        seed = seed0 + bs
        d = _gen_input(bs, seed, dev)
        x = d["hidden_states"]
        idt = d["topk_ids"].to(torch.int32).contiguous()
        tw = d["topk_weights"].float().contiguous()
        outs = {name: torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev) for name in variants}
        fns = {}
        for name, ext in exts.items():
            out = outs[name]

            def fn(ext=ext, out=out):
                ext.okc_run(state["w"], state["scratch"], x, idt, tw, out, 0, 32, state["grid"])

            fns[name] = fn
        graphs = {name: capture(fn) for name, fn in fns.items()}
        torch.cuda.synchronize()

        ref = outs["mid8"]
        same = torch.equal(ref, outs["skt16"])
        rel = float((outs["skt16"].float() - ref.float()).norm() / ref.float().norm())
        print(f"[v161] parity seed={seed} exact={same} rel={rel:.3e}", flush=True)
        assert same and rel == 0.0

        local = idt[(idt >= 0) & (idt < 32)]
        counts = torch.bincount(local, minlength=32)
        touched = int((counts > 0).sum())
        jobs8 = int((((counts + 7) // 8) * (counts > 0)).sum())
        jobs16 = touched
        saved = jobs8 - jobs16

        times = {name: [] for name in variants}
        names = list(variants)
        for rep in range(17):
            order = names if rep & 1 == 0 else list(reversed(names))
            order = order[si % len(order):] + order[:si % len(order)]
            for name in order:
                flush.add_(1)
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[name].replay()
                e1.record()
                torch.cuda.synchronize()
                times[name].append(e0.elapsed_time(e1) * 1000.0)
        med = {name: sorted(ts)[len(ts) // 2] for name, ts in times.items()}
        ratio = med["skt16"] / med["mid8"]
        ratios.append(ratio)
        (saved_ratios if saved else unsplit_ratios).append(ratio)
        wins += ratio < 1.0
        print(
            f"[v161] seed={seed} touched={touched} jobs8={jobs8} jobs16={jobs16} saved={saved} "
            f"mid8={med['mid8']:.1f}us skt16={med['skt16']:.1f}us ratio={ratio:.5f}x",
            flush=True,
        )
        del graphs

    ordered = sorted(ratios)
    median = ordered[len(ordered) // 2]
    mean = sum(ratios) / len(ratios)
    split_mean = sum(saved_ratios) / len(saved_ratios) if saved_ratios else float("nan")
    unsplit_mean = sum(unsplit_ratios) / len(unsplit_ratios) if unsplit_ratios else float("nan")
    passed = wins >= 14 and mean <= 0.995 and median <= 0.997
    print(
        f"[v161] summary wins={wins}/21 mean={mean:.5f}x median={median:.5f}x "
        f"split={len(saved_ratios)} mean={split_mean:.5f}x unsplit={len(unsplit_ratios)} "
        f"mean={unsplit_mean:.5f}x promotion_gate={passed}",
        flush=True,
    )
    print(f"[v161] total {time.time() - t0:.0f}s", flush=True)
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new)


def main() -> None:
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest().upper()
    if digest != EXPECTED:
        raise RuntimeError(f"protected source hash changed: {digest}")
    text = SOURCE.read_text()
    text = replace_once(text, CONFIG_OLD, CONFIG_NEW, "diagnostic config")
    start = text.index("def _diag(state, weights):")
    end = text.index("# Seed variation", start)
    text = text[:start] + DIAG_FUNCTION + "\n" + text[end:]
    TARGET.write_text(text)
    print(f"wrote {TARGET.name}")
    print(f"sha256={hashlib.sha256(TARGET.read_bytes()).hexdigest().upper()}")


if __name__ == "__main__":
    main()
