from pathlib import Path


BASE = Path("submission_plain_exact_v180.py")
OUT = Path("submission_plain_gqx_k64_m256_v184.py")


text = BASE.read_text(encoding="utf-8")

old_configs = '''GQX_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): grid = N / (64 nwg) * cl CTAs
    ("qkv_gate", 128): (1, 3, 8),   # m 33..128: 36 row blocks x 3 K-thirds = 108 CTAs (cuBLAS 16.4 -> 14.0 us at m=128)
}'''
new_configs = '''GQX_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): grid = N / (64 nwg) * cl CTAs
    ("qkv_gate", 128): (1, 3, 8),   # proven route: 108 CTAs
    # v184: retain the original K=64 ring and give each scored m=256 shape
    # 108-128 CTAs.  Unlike v177/v181, this does not double K to 128.
    ("qkv_gate", 256): (1, 3, 4),
    ("pooler", 256): (1, 2, 4),
    ("unpooler", 256): (1, 1, 5),
}'''
if text.count(old_configs) != 1:
    raise RuntimeError("unexpected GQX_CONFIGS anchor count")
text = text.replace(old_configs, new_configs)

old_pick = '''            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]'''
new_pick = '''            best = min(range(len(cands)), key=lambda i: med[i])
            # v184: K=64 GQX is experimental at m=256.  Remove the ~4 us graph
            # replay floor and require a 3% kernel-core win over the best existing
            # route.  This prevents a noisy near-tie from regressing production.
            if m == 256 and layer in ("qkv_gate", "pooler", "unpooler"):
                gqx = [i for i, ch in enumerate(cands) if ch[0] == "gqx"]
                incumbent = [i for i, ch in enumerate(cands) if ch[0] != "gqx"]
                if gqx and incumbent:
                    gi = min(gqx, key=lambda i: med[i])
                    ii = min(incumbent, key=lambda i: med[i])
                    gqx_core = max(1000.0 * med[gi] - 4.0, 1e-6)
                    incumbent_core = max(1000.0 * med[ii] - 4.0, 1e-6)
                    ratio = gqx_core / incumbent_core
                    keep = ratio <= 0.97
                    print("v184 GQX64 gate", layer, m,
                          "candidate", 1000.0 * med[gi],
                          "incumbent", 1000.0 * med[ii],
                          "core_ratio", ratio, "qualified", keep, flush=True)
                    best = gi if keep else ii
            state["choice"][(layer, m)] = cands[best]'''
if text.count(old_pick) != 1:
    raise RuntimeError("unexpected autotune selection anchor count")
text = text.replace(old_pick, new_pick)

text = text.replace("okc_bf16proj_v141_", "okc_bf16proj_v184_", 1)
OUT.write_text(text, encoding="utf-8", newline="\n")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
