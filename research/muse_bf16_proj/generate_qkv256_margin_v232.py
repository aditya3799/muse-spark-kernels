"""Generate v232: QKV/m256 tile-124 cluster-2 margin isolate on v221."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v221.py"
REFERENCE = ROOT / "submission_router32_st8_qkv256_lt124_v146.py"
OUTPUT = ROOT / "submission_plain_qkv256_margin_v232.py"

text = BASE.read_text(encoding="utf-8")
reference = REFERENCE.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match, found {count}: {old[:100]!r}")
    text = text.replace(old, new, 1)


# The exact seeded plan needs the public algo initializer and cluster attribute.
replace_once(
    '            "cublasLtDestroy": [V],\n',
    '            "cublasLtDestroy": [V],\n'
    '            "cublasLtMatmulAlgoInit": [V, I, I, I, I, I, I, I, _ct.POINTER(rq_LtAlgo)],\n',
)

start = "class rq_LtPlan:\n"
end = "\n\nRQ_LT = ["
base_a = text.index(start)
base_b = text.index(end, base_a)
ref_a = reference.index(start)
ref_b = reference.index(end, ref_a)
text = text[:base_a] + reference[ref_a:ref_b] + text[base_b:]

replace_once(
    '    rq = {"t": {}, "plans": {}}\n',
    '    rq = {"t": {}, "plans": {}, "v232_exp": {}}\n',
)

anchor = '''            except Exception:
                pass
    except Exception as e:
'''
insertion = '''            except Exception:
                pass
        try:
            w = weights["qkv_gate"]
            n, k = w.shape
            plan = rq_LtPlan(api, 256, n, k, False, 124, 0, False,
                             cluster=2, seed=True)
            gen = torch.Generator(device=w.device).manual_seed(232124)
            x = torch.randn((256, k), device=w.device,
                            generator=gen).to(torch.bfloat16)
            x[-1].zero_()
            x[-1, 7919] = 1
            out = torch.empty((256, n), dtype=torch.bfloat16, device=w.device)
            if not plan.run(x, w, out):
                raise RuntimeError("cuBLASLt launch rejected")
            ref64 = torch.mm(x[:8].double(), w.double().t())
            torch.cuda.synchronize(w.device)
            err64 = float((out[:8].double() - ref64).norm() /
                          ref64.norm().clamp_min(1e-30))
            probe_exact = bool(torch.equal(out[-1], w[:, 7919]))
            finite = bool(torch.isfinite(out).all())
            if finite and probe_exact and err64 < 5e-3:
                rq["v232_exp"]["qkv124"] = plan
            print("v232 QKV256 plan validation", "fp64", err64,
                  "probe_exact", probe_exact, "finite", finite,
                  "enabled", "qkv124" in rq["v232_exp"], flush=True)
        except Exception as exc:
            print("v232 QKV256 plan unavailable", type(exc).__name__,
                  str(exc)[:300], flush=True)
    except Exception as e:
'''
replace_once(anchor, insertion)

replace_once(
    '''    if (layer, m) in rq.get("plans", {}):
        c.append(("rq_lt",))
    return c
''',
    '''    if (layer, m) in rq.get("plans", {}):
        c.append(("rq_lt",))
    if layer == "qkv_gate" and m == 256 and "qkv124" in rq.get("v232_exp", {}):
        c.append(("v232_qkv124",))
    return c
''',
)

replace_once(
    '''    if choice[0] == "rq_lt":
        plan = rq["plans"].get((layer, m))
''',
    '''    if choice[0] in ("rq_lt", "v232_qkv124"):
        if choice[0] == "v232_qkv124" and (layer != "qkv_gate" or m != 256):
            return _run_torch(x, layer, m, n, k, state)
        plan = (rq.get("v232_exp", {}).get("qkv124")
                if choice[0] == "v232_qkv124"
                else rq["plans"].get((layer, m)))
''',
)

replace_once(
    '    if kind in ("rq_lt", "rq_tma"):\n',
    '    if kind in ("rq_lt", "rq_tma", "v232_qkv124"):\n',
)

replace_once(
    '''            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
''',
    '''            best = min(range(len(cands)), key=lambda i: med[i])
            if layer == "qkv_gate" and m == 256 and cands[best][0] == "v232_qkv124":
                incumbents = [i for i, ch in enumerate(cands)
                              if ch[0] != "v232_qkv124"]
                incumbent = min(incumbents, key=lambda i: med[i])
                candidate_core = max(1000.0 * med[best] - 4.0, 1e-6)
                incumbent_core = max(1000.0 * med[incumbent] - 4.0, 1e-6)
                ratio = candidate_core / incumbent_core
                print("v232 QKV256 margin candidate", 1000.0 * med[best],
                      "incumbent", 1000.0 * med[incumbent],
                      "core_ratio", ratio, flush=True)
                if ratio > 0.99:
                    best = incumbent
            state["choice"][(layer, m)] = cands[best]
''',
)

OUTPUT.write_text(text, encoding="utf-8", newline="\n")
compile(text, str(OUTPUT), "exec")
print(OUTPUT.name, len(text.encode("utf-8")))
