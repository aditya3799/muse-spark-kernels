# muse_decode_layer research workspace

Isolated workspace for the Muse Spark Open whole-decoder-layer challenge.

The sibling repositories are reference-only:

- `../muse_attention_decode`
- `../muse_bf16_proj`
- `../muse_prefill_layer`
- `../prefill_moe_reference`

Do not edit, format, clean, reset, or generate files in those directories from this
repository. In particular, `../muse_prefill_layer` already contains uncommitted work.

## Current state

- The official starter was fetched with `okc init muse_decode_layer` on 2026-10-28.
- `submission.py` is the validated S010/E029 decoder with shape-specific CPB2 large-token MoE dispatch.
- S010 passed 13/13 tests at 1.692448x (13.3654 ms), 1.39% lower latency than S009.
- Live allowance after E033: 12 checks and 8 scored submissions for the current UTC day.
- SGLang baseline: 22.6203 ms per decode step.
- Live decoder rank: 359/905.
- Live decoder leader at the latest snapshot: 2.4029x, or 9.4139 ms.

Read [RESEARCH.md](RESEARCH.md) before implementation and record every remote run in
[EXPERIMENTS.md](EXPERIMENTS.md).

## Files

- `TASK.md`, `AGENTS.md`: official problem contract and workflow.
- `reference.py`, `checks.py`, `nvfp4.py`: judge semantics and tolerances.
- `baseline.py`: SGLang implementation at commit `a55ab7d5`.
- `submission.py`: current S010/E029 candidate and the promoted 13/13 passing source.
- `RESEARCH.md`: workload analysis and optimization roadmap.
- `EXPERIMENTS.md`: run ledger and acceptance gates.

## Commands

From Ubuntu WSL:

```bash
cd /mnt/d/OpenKernelChallenge/muse_decode_layer
../bin/okc check submission.py --json
../bin/okc submit submission.py --json
../bin/okc leaderboard muse_decode_layer --json
```

Always use a check before a scored submission. A candidate must pass all 13 tests,
including page-size 64, scaled KV cache, million-token context, mixed request lengths,
and `T=264` (`batch=33`, decode) before it is eligible for a scored run.

## Working rules

1. Preserve the first line of `submission.py`: `# okc: problem=muse_decode_layer`.
2. Keep weights at their required precision. Repacking in `prepare()` is allowed;
   lossy downcasting is not.
3. Keep `kernel_fn` CUDA-graph safe: no host synchronization, input-dependent Python
   branches, process creation, network access, or cached results derived from an input.
4. Optimize genuine computation. Do not systematically bias routing away from the local
   EP rank merely to exploit routing slack.
5. Judge improvements by paired, cold-L2 CUDA-graph results, not eager local timing.
