"""Generate v194: retest the known Pooler/m256 cuBLASLt heuristic near-win.

v183 measured raw heuristic index 2 at an adjusted 0.9813 ratio but rejected it
under a 3% gate.  This isolated candidate retests exactly that plan on v187 and
accepts it only when the same-run adjusted ratio is at most 0.985.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_qkv1_qkv32_v187.py"
OUTPUT = ROOT / "submission_plain_pooler256_heuristic_v194.py"

text = BASE.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    if text.count(old) != 1:
        raise RuntimeError(f"unexpected anchor count for {old[:100]!r}")
    text = text.replace(old, new, 1)


replace_once(
    "def __init__(self, api, m, n, k, transposed, tile, custom, pick=False):",
    "def __init__(self, api, m, n, k, transposed, tile, custom, pick=False, heuristic_index=None):",
)
replace_once(
    """            result, count = rq_LtResult(), I()
            if pick:
""",
    """            result, count = rq_LtResult(), I()
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
            if pick:
""",
)
replace_once(
    '    rq = {"t": {}, "plans": {}}\n',
    '    rq = {"t": {}, "plans": {}, "v194_plans": {}}\n',
)
replace_once(
    """            except Exception:
                pass
    except Exception as e:
""",
    """            except Exception:
                pass
        try:
            n, k = weights["pooler"].shape
            rq["v194_plans"]["rq_v194_pooler_256_n_02"] = rq_LtPlan(
                api, 256, n, k, False, 0, 0, heuristic_index=2)
        except Exception:
            pass
    except Exception as e:
""",
)
replace_once(
    """    if (layer, m) in rq.get("plans", {}):
        c.append(("rq_lt",))
    return c
""",
    """    if (layer, m) in rq.get("plans", {}):
        c.append(("rq_lt",))
    if layer == "pooler" and m == 256 and "rq_v194_pooler_256_n_02" in rq.get("v194_plans", {}):
        c.append(("rq_v194_pooler_256_n_02",))
    return c
""",
)
replace_once(
    """    if choice[0] == "rq_lt":
        plan = rq["plans"].get((layer, m))
""",
    """    if choice[0] == "rq_lt" or choice[0].startswith("rq_v194_"):
        plan = (rq["plans"].get((layer, m)) if choice[0] == "rq_lt"
                else rq.get("v194_plans", {}).get(choice[0]))
""",
)
replace_once(
    '    if kind in ("rq_lt", "rq_tma"):\n',
    '    if kind in ("rq_lt", "rq_tma") or kind.startswith("rq_v194_"):\n',
)
replace_once(
    """            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
""",
    """            best = min(range(len(cands)), key=lambda i: med[i])
            if cands[best][0].startswith("rq_v194_"):
                incumbents = [i for i, ch in enumerate(cands)
                              if not ch[0].startswith("rq_v194_")]
                incumbent = min(incumbents, key=lambda i: med[i])
                candidate_core = max(1000.0 * med[best] - 4.0, 1e-6)
                incumbent_core = max(1000.0 * med[incumbent] - 4.0, 1e-6)
                ratio = candidate_core / incumbent_core
                print("v194 LT gate", layer, m, cands[best][0],
                      "candidate", 1000.0 * med[best],
                      "incumbent", 1000.0 * med[incumbent],
                      "core_ratio", ratio, flush=True)
                if ratio > 0.985:
                    best = incumbent
            state["choice"][(layer, m)] = cands[best]
""",
)

with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
