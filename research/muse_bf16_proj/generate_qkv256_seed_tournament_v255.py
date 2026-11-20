"""Generate v255: rule-clean QKV/m256 seeded cuBLASLt tournament.

The known tile-124/cluster-2 plan has produced small exact wins, while v221's
incumbent uses tile 134/custom 2.  This isolate exhausts custom options 0..7
and Hopper cluster shapes 2..4 for both tiles.  Every viable plan receives a
one-hot, finite, and FP64 validation before entering the complete-v221 tuner;
promotion additionally requires a one-percent replay-adjusted win.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v221.py"
REFERENCE = ROOT / "submission_router32_st8_qkv256_lt124_v146.py"
OUTPUT = ROOT / "submission_plain_qkv256_seed_tournament_v255.py"

text = BASE.read_text(encoding="utf-8")
reference = REFERENCE.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected one match, found {count}: {old[:100]!r}")
    text = text.replace(old, new, 1)


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
    '    rq = {"t": {}, "plans": {}, "v255_seed_plans": {}}\n',
)

anchor = '''            except Exception:
                pass
    except Exception as e:
'''
insertion = '''            except Exception:
                pass
        w = weights["qkv_gate"]
        n, k = w.shape
        gen = torch.Generator(device=w.device).manual_seed(255124)
        x = torch.randn((256, k), device=w.device,
                        generator=gen).to(torch.bfloat16)
        x[-1].zero_()
        x[-1, 7919] = 1
        ref64 = torch.mm(x[:8].double(), w.double().t())
        for tile in (124, 134):
            for custom in range(8):
                for cluster in range(2, 5):
                    name = "v255_qkv_t%d_c%d_k%d" % (
                        tile, custom, cluster)
                    try:
                        plan = rq_LtPlan(
                            api, 256, n, k, False, tile, custom, False,
                            cluster=cluster, seed=True)
                        out = torch.empty(
                            (256, n), dtype=torch.bfloat16, device=w.device)
                        if not plan.run(x, w, out):
                            raise RuntimeError("cuBLASLt launch rejected")
                        torch.cuda.synchronize(w.device)
                        err64 = float((out[:8].double() - ref64).norm() /
                                      ref64.norm().clamp_min(1e-30))
                        probe_exact = bool(torch.equal(out[-1], w[:, 7919]))
                        finite = bool(torch.isfinite(out).all())
                        if finite and probe_exact and err64 < 5e-3:
                            rq["v255_seed_plans"][name] = plan
                        print("v255 QKV256 seed validation", name,
                              "fp64", err64, "probe_exact", probe_exact,
                              "finite", finite,
                              "enabled", name in rq["v255_seed_plans"],
                              flush=True)
                    except Exception as exc:
                        print("v255 QKV256 seed unavailable", name,
                              type(exc).__name__, str(exc)[:160], flush=True)
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
    if layer == "qkv_gate" and m == 256:
        c.extend((name,) for name in sorted(
            rq.get("v255_seed_plans", {})))
    return c
''',
)

replace_once(
    '''    if choice[0] == "rq_lt":
        plan = rq["plans"].get((layer, m))
''',
    '''    if choice[0] == "rq_lt" or choice[0].startswith("v255_qkv_"):
        if choice[0].startswith("v255_qkv_") and (
                layer != "qkv_gate" or m != 256):
            return _run_torch(x, layer, m, n, k, state)
        plan = (rq.get("v255_seed_plans", {}).get(choice[0])
                if choice[0].startswith("v255_qkv_")
                else rq["plans"].get((layer, m)))
''',
)

replace_once(
    '    if kind in ("rq_lt", "rq_tma"):\n',
    '    if kind in ("rq_lt", "rq_tma") or kind.startswith("v255_qkv_"):\n',
)

replace_once(
    '''            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
''',
    '''            best = min(range(len(cands)), key=lambda i: med[i])
            if cands[best][0].startswith("v255_qkv_"):
                incumbents = [i for i, ch in enumerate(cands)
                              if not ch[0].startswith("v255_qkv_")]
                incumbent = min(incumbents, key=lambda i: med[i])
                candidate_core = max(1000.0 * med[best] - 4.0, 1e-6)
                incumbent_core = max(1000.0 * med[incumbent] - 4.0, 1e-6)
                ratio = candidate_core / incumbent_core
                print("v255 QKV256 seed margin", cands[best][0],
                      "candidate", 1000.0 * med[best],
                      "incumbent", 1000.0 * med[incumbent],
                      "core_ratio", ratio, flush=True)
                if ratio > 0.99:
                    best = incumbent
            state["choice"][(layer, m)] = cands[best]
''',
)

compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
