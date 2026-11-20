"""Build an integrated q1 occ3 + conservative q8 F2 candidate.

The F2 consumers dynamically claim tiles.  A fast warpgroup can claim tile
``i + STAGES`` before the producer has recycled tile ``i``'s stage.  Waiting on
that future mbarrier parity is unsafe: while the barrier still has the older
parity, ``try_wait.parity`` can treat the requested phase as already complete.
The per-stage generation below makes the producer publish the tile number only
after the prior empty phase has completed, before any consumer waits on the
new full phase.  With OKC_XP_ARRALL enabled, F2's empty barrier must count all
256 threads in the two participating warpgroups (rather than the stale count
of eight warp leaders); this preserves the all-lane stage-lifetime guarantee.
The two streams reduce independently through the proven FP32 workspace path,
and the flat grid uses the proven last-CTA finalizer instead of the new
512-consumer cooperative finalizer.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_q1_occ3_np128.py"
DESTINATION = ROOT / "submission_q1_occ3_q8_f2_safe.py"


REPLACEMENTS = (
    (
        "#ifndef OKC_NP8\n#define OKC_NP8 128     // q_len 8: 4 producer warps (its consumers need the registers)\n#endif",
        "#ifndef OKC_NP8\n"
        "#define OKC_NP8 128     // q_len 8 F2: original single producer warpgroup\n"
        "#endif",
    ),
    (
        "#define OKC_F2 0        // q_len 8 flat: one CTA per SM, two streams claiming tiles per KV head (measured slower: one producer warpgroup per SM limits)",
        "#define OKC_F2 1        // q_len 8 flat: one CTA per SM, guarded two-stream tile claiming",
    ),
    (
        "  static constexpr bool F2 = (L == 8) && F && OKC_F2 && OKC_SM90 && PV8ON && OKC_PV8FLAT && OKC_QSS && !OKC_CONT && !OKC_PTR;\n"
        "  static constexpr int NT = (L == 1) ? 128 : ((S2 || F2) ? 512 : 256);",
        "  static constexpr bool F2 = (L == 8) && F && OKC_F2 && OKC_SM90 && PV8ON && OKC_PV8FLAT && OKC_QSS && !OKC_PTR;\n"
        "  // q1 keeps its continuous cross-segment pipeline; F2 q8 drains each segment.\n"
        "  static constexpr bool CONT = (bool)OKC_CONT && !F2;\n"
        "  static constexpr int NT = (L == 1) ? 128 : ((S2 || F2) ? 512 : 256);",
    ),
    (
        "  constexpr bool CONT = FLAT && Cfg<L, FLAT>::WS && OKC_CONT;",
        "  constexpr bool CONT = FLAT && Cfg<L, FLAT>::WS && Cfg<L, FLAT>::CONT;",
    ),
    (
        "  constexpr bool PTR = (L == 8) && FLAT && Cfg<L, FLAT>::WS && OKC_PTR && PV8ON && OKC_PV8FLAT && OKC_QSS && !OKC_CONT;",
        "  constexpr bool PTR = (L == 8) && FLAT && Cfg<L, FLAT>::WS && OKC_PTR && PV8ON && OKC_PV8FLAT && OKC_QSS && !CONT;",
    ),
    (
        "  uint64_t mbar[8];   // TMA mode: one per stage (64 bytes keeps the members after it 16-byte aligned)\n",
        "  uint64_t mbar[8];   // TMA mode: one per stage (64 bytes keeps the members after it 16-byte aligned)\n"
        "  int gen[8];         // F2 stage generation; separate from the generic TMA barriers\n",
    ),
    (
        "          mbar_init_n(&S.empty[st], Cfg<L, FLAT>::F2 ? 8 : (OKC_XP_ARRALL ? NT : NT / 32));   // F2: one warpgroup per head takes each tile\n"
        "        }",
        "          mbar_init_n(&S.empty[st], Cfg<L, FLAT>::F2 ? (OKC_XP_ARRALL ? NT / 2 : 8) : (OKC_XP_ARRALL ? NT : NT / 32));   // F2: one warpgroup per head takes each tile\n"
        "          if constexpr (Cfg<L, FLAT>::F2) S.gen[st] = -1;\n"
        "        }",
    ),
    (
        "        if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);\n"
        "        PROF_MARK(0);",
        "        if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);\n"
        "        // Do not let an F2 consumer issue a future-parity wait before this stage is recycled.\n"
        "        if constexpr (Cfg<L, FLAT>::F2) {\n"
        "          if (q == 0) reinterpret_cast<volatile int*>(S.gen)[st] = x;\n"
        "        }\n"
        "        PROF_MARK(0);",
    ),
    (
        "      const int st0 = (tb + i) % STAGES;\n"
        "      mbar_wait_full(&S.full[st0], ((tb + i) / STAGES) & 1);   // tile i landed",
        "      const int st0 = (tb + i) % STAGES;\n"
        "      // A future parity is ambiguous until the producer has advanced the stage generation.\n"
        "      while (reinterpret_cast<volatile int*>(S.gen)[st0] != i) {}\n"
        "      mbar_wait_full(&S.full[st0], ((tb + i) / STAGES) & 1);   // tile i landed",
    ),
    (
        "  int i_cur = 0, cnt2 = 0;   // F2: this warpgroup's claimed tile, and how many it has taken\n"
        "  if constexpr (F2) {\n"
        "    if (tid == 0) { S.claim[0] = 0; S.claim[1] = 0; }\n"
        "    csync<WS, NT>();   // the q staging (S.q8s) is complete for both streams; counters reset\n"
        "    if ((tid & 127) == 0) S.nxt[wgi][0] = atomicAdd(&S.claim[hk], 1);\n"
        "    wgsync(wgi);\n"
        "    i_cur = S.nxt[wgi][0];\n"
        "  }\n"
        "  const int nit = S2 ? ((ntiles + 1) >> 1) : (F2 ? 0x7fffffff : ntiles);",
        "  int cnt2 = 0;\n"
        "  if constexpr (F2) csync<WS, NT>();   // q staging is complete before either deterministic stream starts\n"
        "  const int nit = (S2 || F2) ? ((ntiles + 1) >> 1) : ntiles;",
    ),
    (
        "    const int i = S2 ? 2 * jj + sid : (F2 ? i_cur : jj);   // S2: tiles 2jj and 2jj + 1; F2: claimed",
        "    const int i = (S2 || F2) ? 2 * jj + sid : jj;   // deterministic even/odd tile streams",
    ),
    (
        "      // the next claim, read after this tile's warpgroup barrier\n"
        "      if ((tid & 127) == 0) S.nxt[wgi][(cnt2 + 1) & 1] = atomicAdd(&S.claim[hk], 1);\n"
        "      const int st0 = (tb + i) % STAGES;",
        "      const int st0 = (tb + i) % STAGES;",
    ),
    (
        "    if constexpr (F2) {\n"
        "      i_cur = S.nxt[wgi][(cnt2 + 1) & 1];   // written before this tile's warpgroup barrier\n"
        "      ++cnt2;\n"
        "    }",
        "    if constexpr (F2) ++cnt2;",
    ),
    (
        "  if constexpr (F2) {   // stream 1's warpgroup of each head hands its state to stream 0's (same rows)",
        "  if constexpr (F2 && false) {   // keep both stream states separate; the FP32 workspace combines them",
    ),
    (
        "  if constexpr (Cfg<L, FLAT>::F2) do_red = (warp >> 3) == 0;   // stream 1's state was merged into stream 0",
        "  if constexpr (Cfg<L, FLAT>::F2) do_red = true;   // both streams independently reduce through the proven path",
    ),
    (
        "    if constexpr (PV8ON && OKC_QSS && (!FLAT || OKC_PV8FLAT)) {   // the A tile for the SS QK^T wgmma\n"
        "#pragma unroll\n"
        "      for (int ks = 0; ks < 2; ++ks)\n"
        "#pragma unroll\n"
        "        for (int hf = 0; hf < 2; ++hf) {",
        "    if constexpr (PV8ON && OKC_QSS && (!FLAT || OKC_PV8FLAT)) {   // the A tile for the SS QK^T wgmma\n"
        "      if (!F2 || sid == 0) {   // F2 streams share q8s: only stream 0 publishes it\n"
        "#pragma unroll\n"
        "      for (int ks = 0; ks < 2; ++ks)\n"
        "#pragma unroll\n"
        "        for (int hf = 0; hf < 2; ++hf) {",
    ),
    (
        "        }\n"
        "      fence_proxy_async();   // read by the wgmma after the loop's first barrier\n"
        "    }\n"
        "    if constexpr (PTR)",
        "        }\n"
        "      fence_proxy_async();   // read by the wgmma after the loop's first barrier\n"
        "      }\n"
        "    }\n"
        "    if constexpr (PTR)",
    ),
    (
        "if constexpr (OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !OKC_CONT && !OKC_PTR)",
        "if constexpr (OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !Cfg<L, FLAT>::CONT && !OKC_PTR)",
    ),
    (
        "mbar_init_n(&S.empty[st], Cfg<L, FLAT>::F2 ? 8 : (OKC_XP_ARRALL ? Cfg<L, FLAT>::NT : Cfg<L, FLAT>::NT / 32));",
        "mbar_init_n(&S.empty[st], Cfg<L, FLAT>::F2 ? (OKC_XP_ARRALL ? Cfg<L, FLAT>::NT / 2 : 8) : (OKC_XP_ARRALL ? Cfg<L, FLAT>::NT : Cfg<L, FLAT>::NT / 32));",
    ),
    (
        "if constexpr (Cfg<L, FLAT>::WS && OKC_CONT) {   // continuous pipeline: stage barriers set up once per launch",
        "if constexpr (Cfg<L, FLAT>::WS && Cfg<L, FLAT>::CONT) {   // continuous pipeline: stage barriers set up once per launch",
    ),
    (
        "                            OKC_CONT ? tb : 0, 0,\n"
        "                            OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !OKC_CONT && !OKC_PTR && k == 0);",
        "                            Cfg<L, FLAT>::CONT ? tb : 0, 0,\n"
        "                            OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !Cfg<L, FLAT>::CONT && !OKC_PTR && k == 0);",
    ),
    (
        "    if constexpr (R == 1 && OKC_CONT) cp_wait<0>();   // the producer leaves with its copies landed",
        "    if constexpr (R == 1 && Cfg<L, FLAT>::CONT) cp_wait<0>();   // the producer leaves with its copies landed",
    ),
    (
        'name="okc_attn_v298_q1occ3_np128_90a"',
        'name="okc_attn_v298_q1occ3_q8f2safe_90a"',
    ),
    (
        "OKC_FLATCOOP = 1",
        "OKC_FLATCOOP = 0   # F2 uses the proven per-request last-CTA finalizer",
    ),
    (
        "OKC_F2 = 0   # must match the CUDA macro OKC_F2 (q_len 8 flat: 1 CTA per SM, two streams)",
        "OKC_F2 = 1   # must match the guarded q_len 8 F2 CUDA path",
    ),
)


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"expected one anchor, found {count}: {old[:100]!r}")
        text = text.replace(old, new, 1)
    with DESTINATION.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(DESTINATION)


if __name__ == "__main__":
    main()
