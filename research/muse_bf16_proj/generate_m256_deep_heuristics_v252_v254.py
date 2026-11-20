"""Generate deep zero-workspace cuBLASLt m=256 isolates on complete v221.

v183/v228 cover heuristic indices 0..11 in both weight layouts.  These three
isolates cover the previously untouched indices 12..47 for Pooler, QKV, and
Unpooler independently.  A route is promoted only after a one-percent
replay-overhead-adjusted win over the fastest complete-v221 incumbent.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v221.py"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected one match, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def build(version: int, layer: str, start: int = 12, stop: int = 48,
          request: int = 48, label: str = "deep_heuristics") -> None:
    text = BASE.read_text(encoding="utf-8")
    tag = f"v{version}"
    key = f"{tag}_heuristic_plans"
    prefix = f"{tag}_h_"

    text = replace_once(
        text,
        "def __init__(self, api, m, n, k, transposed, tile, custom, pick=False):",
        "def __init__(self, api, m, n, k, transposed, tile, custom, pick=False, heuristic_index=None):",
    )
    text = replace_once(
        text,
        '''            result, count = rq_LtResult(), I()
            if pick:
''',
        f'''            result, count = rq_LtResult(), I()
            if heuristic_index is not None:
                many = (rq_LtResult * {request})()
                check(lib.cublasLtMatmulAlgoGetHeuristic(
                    api.handle, self.desc, self.a, self.b, self.c, self.c,
                    pref, {request}, many, _ct.byref(count)))
                viable = [i for i in range(count.value)
                          if not many[i].state and not many[i].workspace]
                if heuristic_index >= len(viable):
                    raise RuntimeError("No zero-workspace heuristic at index %d" % heuristic_index)
                selected = many[viable[heuristic_index]].algo
                checked = rq_LtResult()
                check(lib.cublasLtMatmulAlgoCheck(
                    api.handle, self.desc, self.a, self.b, self.c, self.c,
                    _ct.byref(selected), _ct.byref(checked)))
                check(checked.state)
                if checked.workspace:
                    raise RuntimeError("Heuristic unexpectedly needs workspace")
                self.algo = selected
                return
            if pick:
''',
    )
    text = replace_once(
        text,
        '''RQ_TMA = {("pooler", 128): (64, 64, 128, 6), ("pooler", 256): (64, 64, 128, 6),
          ("unpooler", 128): (64, 128, 128, 4), ("unpooler", 256): (128, 128, 64, 6)}
''',
        f'''RQ_TMA = {{("pooler", 128): (64, 64, 128, 6), ("pooler", 256): (64, 64, 128, 6),
          ("unpooler", 128): (64, 128, 128, 4), ("unpooler", 256): (128, 128, 64, 6)}}
_{tag.upper()}_ROUTES = (("{layer}", 256),)
_{tag.upper()}_HEURISTIC_START = {start}
_{tag.upper()}_HEURISTIC_STOP = {stop}
''',
    )
    text = replace_once(
        text,
        '    rq = {"t": {}, "plans": {}}\n',
        f'    rq = {{"t": {{}}, "plans": {{}}, "{key}": {{}}}}\n',
    )
    text = replace_once(
        text,
        '''            except Exception:
                pass
    except Exception as e:
''',
        f'''            except Exception:
                pass
        for layer, m in _{tag.upper()}_ROUTES:
            n, k = weights[layer].shape
            for transposed in (False, True):
                for index in range(_{tag.upper()}_HEURISTIC_START,
                                   _{tag.upper()}_HEURISTIC_STOP):
                    name = "{prefix}%s_%d_%s_%02d" % (
                        layer, m, "t" if transposed else "n", index)
                    try:
                        rq["{key}"][name] = rq_LtPlan(
                            api, m, n, k, transposed, 0, 0,
                            heuristic_index=index)
                    except Exception:
                        pass
    except Exception as e:
''',
    )
    text = replace_once(
        text,
        '''    if (layer, m) in rq.get("plans", {}):
        c.append(("rq_lt",))
    return c
''',
        f'''    if (layer, m) in rq.get("plans", {{}}):
        c.append(("rq_lt",))
    prefix = "{prefix}%s_%d_" % (layer, m)
    c.extend((name,) for name in sorted(rq.get("{key}", {{}}))
             if name.startswith(prefix))
    return c
''',
    )
    text = replace_once(
        text,
        '''    if choice[0] == "rq_lt":
        plan = rq["plans"].get((layer, m))
''',
        f'''    if choice[0] == "rq_lt" or choice[0].startswith("{prefix}"):
        plan = (rq["plans"].get((layer, m)) if choice[0] == "rq_lt"
                else rq.get("{key}", {{}}).get(choice[0]))
''',
    )
    text = replace_once(
        text,
        '    if kind in ("rq_lt", "rq_tma"):\n',
        f'    if kind in ("rq_lt", "rq_tma") or kind.startswith("{prefix}"):\n',
    )
    text = replace_once(
        text,
        '''            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
''',
        f'''            best = min(range(len(cands)), key=lambda i: med[i])
            if cands[best][0].startswith("{prefix}"):
                incumbents = [i for i, ch in enumerate(cands)
                              if not ch[0].startswith("{prefix}")]
                incumbent = min(incumbents, key=lambda i: med[i])
                candidate_core = max(1000.0 * med[best] - 4.0, 1e-6)
                incumbent_core = max(1000.0 * med[incumbent] - 4.0, 1e-6)
                ratio = candidate_core / incumbent_core
                print("{tag} deep LT gate", layer, m, cands[best][0],
                      "candidate", 1000.0 * med[best],
                      "incumbent", 1000.0 * med[incumbent],
                      "core_ratio", ratio, flush=True)
                if ratio > 0.99:
                    best = incumbent
            state["choice"][(layer, m)] = cands[best]
''',
    )

    output = ROOT / f"submission_plain_{layer}256_{label}_{tag}.py"
    compile(text, str(output), "exec")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    print(output.name, len(text.encode("utf-8")))


for candidate_version, candidate_layer in (
        (252, "pooler"), (253, "qkv_gate"), (254, "unpooler")):
    build(candidate_version, candidate_layer)

for candidate_version, candidate_layer in (
        (268, "pooler"), (269, "qkv_gate"), (270, "unpooler")):
    build(candidate_version, candidate_layer, start=48, stop=96,
          request=96, label="tail_heuristics")
