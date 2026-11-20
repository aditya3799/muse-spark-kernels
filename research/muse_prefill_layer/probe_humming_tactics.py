# okc: problem=muse_moe_w4a16
"""Untimed H200 probe for the Humming configurations used by muse_prefill_layer.

The routed-expert problem supplies the same 4096-wide NVFP4 expert weights and has
independent check quota.  prepare() benchmarks CUDA-graph replays and deliberately
stops before correctness/timing; kernel_fn is therefore only a placeholder.
"""

import json

import torch
from humming.config import GemmType
from humming.layer import HummingLayer, HummingMethod
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size


LOCAL_EXPERTS = 32
TOP_K = 8
K = 4096
W13_N = 8192
W2_N = 4096
HUGE_M = 1 << 30
REPEATS = 5


def _make_layer(n, packed, scale, global_scale):
    config = {
        "dtype": "float4e2m1",
        "scale_dtype": "float8e4m3",
        "group_size": 16,
        "scale_2_type": "tensor",
    }
    layer = HummingLayer(
        shape_n=n,
        shape_k=K,
        weight_config=config,
        num_experts=LOCAL_EXPERTS,
        torch_dtype=torch.bfloat16,
    ).cuda()
    with torch.no_grad():
        layer.weight.copy_(packed[:LOCAL_EXPERTS].contiguous().view(torch.int32))
        layer.weight_scale.copy_(scale[:LOCAL_EXPERTS])
        layer.weight_scale_2.copy_(global_scale[:LOCAL_EXPERTS])
    layer.transform()
    layer.eval()
    return layer


def _configs_by_block(layer):
    configs = HummingMethod.get_default_tuning_configs(
        layer=layer, use_f16_accum=False, gemm_type=GemmType.INDEXED
    )
    by_block = {}
    for _, _, config in configs:
        by_block.setdefault(config["block_shape"][0], config)
    return configs, by_block


def _forced(config):
    return json.dumps([[0, HUGE_M, config]])


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = fn()
    torch.cuda.synchronize()
    return graph, result, stream


def _time_graph(graph, flush):
    samples = []
    for _ in range(REPEATS):
        flush.add_(1)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def _selected_block(configs, valid_shape_m):
    for min_m, max_m, config in configs:
        if valid_shape_m > min_m and valid_shape_m <= max_m:
            return config["block_shape"][0]
    raise RuntimeError(f"no default config for M={valid_shape_m}")


def prepare(weights):
    w13 = _make_layer(W13_N, weights["w13"], weights["w13_sf"], weights["w13_gs"])
    w2 = _make_layer(W2_N, weights["w2"], weights["w2_sf"], weights["w2_gs"])
    w13_defaults, w13_by_block = _configs_by_block(w13)
    w2_defaults, w2_by_block = _configs_by_block(w2)
    common_blocks = set(w13_by_block).intersection(w2_by_block)
    compute = json.dumps({"gemm_type": "indexed", "use_f16_accum": False})
    flush = torch.empty(96 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    cases = (
        ("B1Q753", 753, 1017, (16, 24, 32, 40, 48, 56, 64, 72, 80)),
        ("B4Q753", 3012, 3108, (64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144)),
        ("B1Q16K", 16384, 16512, (128, 136, 144, 152, 160, 168, 176)),
    )
    print("TACTIC_SWEEP units=us timing=cold-L2 CUDA-graph median5")
    for name, tokens, valid_m, requested_blocks in cases:
        routes = tokens * TOP_K
        x = torch.randn((tokens, K), dtype=torch.bfloat16, device="cuda")
        local_ids = torch.full((tokens, TOP_K), -1, dtype=torch.int32, device="cuda")
        # One local route per token, evenly spread over the 32 local experts.
        local_ids[:, 0] = torch.arange(tokens, dtype=torch.int32, device="cuda") % LOCAL_EXPERTS
        gate_up = torch.empty((routes, W13_N), dtype=torch.bfloat16, device="cuda")
        inter = torch.empty((routes, W2_N), dtype=torch.bfloat16, device="cuda")
        y = torch.empty((routes, W2_N), dtype=torch.bfloat16, device="cuda")
        default_block = _selected_block(w13_defaults, valid_m)
        results = []

        # SiLU is independent of the alignment tile and is measured once per shape.
        silu_graph, _, silu_stream = _capture(
            lambda: silu_and_mul(
                gate_up, inter, expert_ids=local_ids.view(-1), expert_step=1
            )
        )
        silu_us = _time_graph(silu_graph, flush)
        del silu_graph, silu_stream

        for block_m in requested_blocks:
            if block_m not in common_blocks:
                continue
            sorted_ids, block_experts, n_padded = moe_align_block_size(
                local_ids,
                block_m,
                LOCAL_EXPERTS,
                ignore_invalid_expert=True,
            )
            padded = int(n_padded.item())
            base_args = dict(
                sorted_ids=sorted_ids,
                expert_ids=block_experts,
                num_tokens_padded=n_padded,
                valid_shape_m=valid_m,
                compute_config=compute,
            )
            w13_tuning = _forced(w13_by_block[block_m])
            w2_tuning = _forced(w2_by_block[block_m])

            g13, _, s13 = _capture(
                lambda: w13(
                    x,
                    outputs=gate_up,
                    top_k=TOP_K,
                    tuning_config=w13_tuning,
                    **base_args,
                )
            )
            w13_us = _time_graph(g13, flush)
            del g13, s13

            g2, _, s2 = _capture(
                lambda: w2(
                    inter,
                    outputs=y,
                    top_k=1,
                    tuning_config=w2_tuning,
                    **base_args,
                )
            )
            w2_us = _time_graph(g2, flush)
            del g2, s2
            total = w13_us + silu_us + w2_us
            results.append((total, block_m, padded, w13_us, w2_us))
            mark = "*" if block_m == default_block else " "
            print(
                f"{name} {mark}bm={block_m:3d} pad={padded:6d} "
                f"w13={w13_us:8.2f} silu={silu_us:7.2f} "
                f"w2={w2_us:8.2f} sum={total:8.2f}"
            )
            torch.cuda.empty_cache()

        results.sort()
        best = results[0]
        current = next(row for row in results if row[1] == default_block)
        gain = 100.0 * (current[0] / best[0] - 1.0)
        print(
            f"WINNER {name} default_bm={default_block} default={current[0]:.2f} "
            f"best_bm={best[1]} best={best[0]:.2f} gain={gain:.2f}%"
        )
        del x, local_ids, gate_up, inter, y
        torch.cuda.empty_cache()

    raise RuntimeError("Humming tactic sweep complete")


def kernel_fn(data, state):
    return torch.empty_like(data["hidden_states"])
