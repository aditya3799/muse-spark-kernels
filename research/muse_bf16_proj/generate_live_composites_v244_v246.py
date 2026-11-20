"""Generate every nontrivial subset of live v228/v229/v232 mechanisms.

v230 is deliberately excluded: static comparison proved that v221 already owns
its exact O/m256 source and configuration.  v244-v246 are staged only for use
after their isolated checks select the corresponding routes.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
V228 = ROOT / "submission_plain_pooler256_margin_v228.py"
V229 = ROOT / "submission_plain_unpool256_cudnn_v229.py"
V232 = ROOT / "submission_plain_qkv256_margin_v232.py"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def unpool_tail() -> str:
    source = V229.read_text(encoding="utf-8")
    marker = "# v229: test FlashInfer's independent cuDNN BF16 engine stack"
    if source.count(marker) != 1:
        raise RuntimeError("v229 marker mismatch")
    return marker + source.split(marker, 1)[1]


def pool_qkv_source() -> str:
    text = V228.read_text(encoding="utf-8")
    qkv = V232.read_text(encoding="utf-8")

    text = replace_once(
        text,
        '            "cublasLtDestroy": [V],\n',
        '            "cublasLtDestroy": [V],\n'
        '            "cublasLtMatmulAlgoInit": [V, I, I, I, I, I, I, I, _ct.POINTER(rq_LtAlgo)],\n',
    )

    start = "class rq_LtPlan:\n"
    end = "\n\nRQ_LT = ["
    qa = qkv.index(start)
    qb = qkv.index(end, qa)
    klass = qkv[qa:qb]
    klass = replace_once(
        klass,
        "def __init__(self, api, m, n, k, transposed, tile, custom, pick=False, cluster=0, seed=False):",
        "def __init__(self, api, m, n, k, transposed, tile, custom, pick=False, cluster=0, seed=False, heuristic_index=None):",
    )
    klass = replace_once(
        klass,
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
    a = text.index(start)
    b = text.index(end, a)
    text = text[:a] + klass + text[b:]

    text = replace_once(
        text,
        '    rq = {"t": {}, "plans": {}, "v228_heuristic_plans": {}}\n',
        '    rq = {"t": {}, "plans": {}, "v228_heuristic_plans": {}, "v232_exp": {}}\n',
    )
    anchor = '''                    except Exception:
                        pass
    except Exception as e:
'''
    insertion = '''                    except Exception:
                        pass
        try:
            w = weights["qkv_gate"]
            n, k = w.shape
            plan = rq_LtPlan(api, 256, n, k, False, 124, 0, False,
                             cluster=2, seed=True)
            gen = torch.Generator(device=w.device).manual_seed(232124)
            x = torch.randn((256, k), device=w.device,
                            generator=gen).to(torch.bfloat16)
            x[-1].zero_(); x[-1, 7919] = 1
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
            print("v244 QKV256 validation", err64, probe_exact, finite,
                  "enabled", "qkv124" in rq["v232_exp"], flush=True)
        except Exception as exc:
            print("v244 QKV256 unavailable", type(exc).__name__,
                  str(exc)[:300], flush=True)
    except Exception as e:
'''
    text = replace_once(text, anchor, insertion)

    text = replace_once(
        text,
        '''    c.extend((name,) for name in sorted(rq.get("v228_heuristic_plans", {}))
             if name.startswith(prefix))
    return c
''',
        '''    c.extend((name,) for name in sorted(rq.get("v228_heuristic_plans", {}))
             if name.startswith(prefix))
    if layer == "qkv_gate" and m == 256 and "qkv124" in rq.get("v232_exp", {}):
        c.append(("v232_qkv124",))
    return c
''',
    )
    text = replace_once(
        text,
        '''    if choice[0] == "rq_lt" or choice[0].startswith("v228_h_"):
        plan = (rq["plans"].get((layer, m)) if choice[0] == "rq_lt"
                else rq.get("v228_heuristic_plans", {}).get(choice[0]))
''',
        '''    if (choice[0] in ("rq_lt", "v232_qkv124") or
            choice[0].startswith("v228_h_")):
        if choice[0] == "v232_qkv124" and (layer != "qkv_gate" or m != 256):
            return _run_torch(x, layer, m, n, k, state)
        if choice[0] == "rq_lt":
            plan = rq["plans"].get((layer, m))
        elif choice[0] == "v232_qkv124":
            plan = rq.get("v232_exp", {}).get("qkv124")
        else:
            plan = rq.get("v228_heuristic_plans", {}).get(choice[0])
''',
    )
    text = replace_once(
        text,
        '    if kind in ("rq_lt", "rq_tma") or kind.startswith("v228_h_"):\n',
        '    if kind in ("rq_lt", "rq_tma", "v232_qkv124") or kind.startswith("v228_h_"):\n',
    )
    text = replace_once(
        text,
        '''            best = min(range(len(cands)), key=lambda i: med[i])
            if cands[best][0].startswith("v228_h_"):
''',
        '''            best = min(range(len(cands)), key=lambda i: med[i])
            if (layer == "qkv_gate" and m == 256 and
                    cands[best][0] == "v232_qkv124"):
                incumbents = [i for i, ch in enumerate(cands)
                              if ch[0] != "v232_qkv124"]
                incumbent = min(incumbents, key=lambda i: med[i])
                candidate_core = max(1000.0 * med[best] - 4.0, 1e-6)
                incumbent_core = max(1000.0 * med[incumbent] - 4.0, 1e-6)
                ratio = candidate_core / incumbent_core
                print("v244 QKV256 gate", 1000.0 * med[best],
                      1000.0 * med[incumbent], ratio, flush=True)
                if ratio > 0.99:
                    best = incumbent
            if cands[best][0].startswith("v228_h_"):
''',
    )
    return text


def emit(name: str, source: str) -> None:
    output = ROOT / name
    compile(source, str(output), "exec")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(source)
    print(output.name, len(source.encode("utf-8")))


pool_qkv = pool_qkv_source().rstrip() + "\n"
tail = unpool_tail().rstrip()
emit("submission_plain_pool256_qkv256_margin_v244.py", pool_qkv)
emit("submission_plain_pool256_qkv256_unpool256_margin_v245.py",
     pool_qkv.rstrip() + "\n\n\n" + tail + "\n")
emit("submission_plain_qkv256_unpool256_margin_v246.py",
     V232.read_text(encoding="utf-8").rstrip() + "\n\n\n" + tail + "\n")
