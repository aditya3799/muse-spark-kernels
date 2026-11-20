"""Generate v183: exact-shape zero-workspace cuBLASLt heuristic tournament.

The existing wide routes keep one hand-seeded algo-66 plan.  This candidate
also tests the first twelve valid zero-workspace heuristic plans for both raw
and pre-transposed weights at the three scored m=256 shapes that still reach
the main dispatcher.  A heuristic plan can enter dispatch only after a 3%
replay-overhead-adjusted cold-L2 win over every incumbent candidate.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_composite_v182.py"
OUTPUT = ROOT / "submission_plain_lt_heuristics_v183.py"

text = BASE.read_text()


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match, found {count}: {old[:100]!r}")
    text = text.replace(old, new, 1)


replace_once(
    "def __init__(self, api, m, n, k, transposed, tile, custom, pick=False, cluster=0, seed=False):",
    "def __init__(self, api, m, n, k, transposed, tile, custom, pick=False, cluster=0, seed=False, heuristic_index=None):",
)

replace_once(
    '''            result, count = rq_LtResult(), I()
            if seed:
''',
    '''            result, count = rq_LtResult(), I()
            if heuristic_index is not None:
                many = (rq_LtResult * 48)()
                check(lib.cublasLtMatmulAlgoGetHeuristic(
                    api.handle, self.desc, self.a, self.b, self.c, self.c,
                    pref, 48, many, _ct.byref(count)))
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
            if seed:
''',
)

replace_once(
    '''RQ_TMA = {("pooler", 128): (64, 64, 128, 6), ("pooler", 256): (64, 64, 128, 6),
          ("unpooler", 128): (64, 128, 128, 4), ("unpooler", 256): (128, 128, 64, 6)}
''',
    '''RQ_TMA = {("pooler", 128): (64, 64, 128, 6), ("pooler", 256): (64, 64, 128, 6),
          ("unpooler", 128): (64, 128, 128, 4), ("unpooler", 256): (128, 128, 64, 6)}
_V183_ROUTES = (("qkv_gate", 256), ("pooler", 256), ("unpooler", 256))
_V183_HEURISTICS = 12
''',
)

replace_once(
    '    rq = {"t": {}, "plans": {}}\n',
    '    rq = {"t": {}, "plans": {}, "heuristic_plans": {}}\n',
)

anchor = '''            except Exception:
                pass
    except Exception as e:
'''
replacement = '''            except Exception:
                pass
        for layer, m in _V183_ROUTES:
            n, k = weights[layer].shape
            for transposed in (False, True):
                for index in range(_V183_HEURISTICS):
                    name = "rq_h_%s_%d_%s_%02d" % (
                        layer, m, "t" if transposed else "n", index)
                    try:
                        rq["heuristic_plans"][name] = rq_LtPlan(
                            api, m, n, k, transposed, 0, 0,
                            heuristic_index=index)
                    except Exception:
                        pass
    except Exception as e:
'''
replace_once(anchor, replacement)

replace_once(
    '''    if (layer, m) in rq.get("plans", {}):
        c.append(("rq_lt",))
    return c
''',
    '''    if (layer, m) in rq.get("plans", {}):
        c.append(("rq_lt",))
    prefix = "rq_h_%s_%d_" % (layer, m)
    c.extend((name,) for name in sorted(rq.get("heuristic_plans", {}))
             if name.startswith(prefix))
    return c
''',
)

replace_once(
    '''    if choice[0] == "rq_lt":
        plan = rq["plans"].get((layer, m))
''',
    '''    if choice[0] == "rq_lt" or choice[0].startswith("rq_h_"):
        plan = (rq["plans"].get((layer, m)) if choice[0] == "rq_lt"
                else rq.get("heuristic_plans", {}).get(choice[0]))
''',
)

replace_once(
    '    if kind in ("rq_lt", "rq_tma"):\n',
    '    if kind in ("rq_lt", "rq_tma") or kind.startswith("rq_h_"):\n',
)

replace_once(
    '''            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
''',
    '''            best = min(range(len(cands)), key=lambda i: med[i])
            if cands[best][0].startswith("rq_h_"):
                incumbents = [i for i, ch in enumerate(cands)
                              if not ch[0].startswith("rq_h_")]
                incumbent = min(incumbents, key=lambda i: med[i])
                candidate_core = max(1000.0 * med[best] - 4.0, 1e-6)
                incumbent_core = max(1000.0 * med[incumbent] - 4.0, 1e-6)
                ratio = candidate_core / incumbent_core
                print("v183 LT gate", layer, m, cands[best][0],
                      "candidate", 1000.0 * med[best],
                      "incumbent", 1000.0 * med[incumbent],
                      "core_ratio", ratio, flush=True)
                if ratio > 0.97:
                    best = incumbent
            state["choice"][(layer, m)] = cands[best]
''',
)

compile(text, str(OUTPUT), "exec")
OUTPUT.write_text(text)
print(OUTPUT.name, len(text.encode()))
