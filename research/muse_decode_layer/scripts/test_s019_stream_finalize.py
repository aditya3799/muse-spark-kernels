"""Structural and reference-model checks for S019 streamed MoE finalization."""

from __future__ import annotations

import ast
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "submission_s019_stream_finalize.py"
PROTECTED = ROOT / "submission.py"
PROTECTED_SHA256 = "42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69"

M = 256
TOPK = 8
NLOC = 32
TPX = 64
NRB2 = 16
NCW = 16
PASS_TARGET = NRB2 * NCW
TASKS_PER_TOKEN = 16


def extract_cuda(text: str) -> str:
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "M_CUDA_X" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("M_CUDA_X not found")


def check_structure() -> None:
    text = CANDIDATE.read_text()
    source = extract_cuda(text)
    assert hashlib.sha256(PROTECTED.read_bytes()).hexdigest() == PROTECTED_SHA256
    assert '128: ("wx", 0, 0, "-DNCW=8 -DRPW=2 -DTPX=32 -DNSL=2 -DCPB=2 -DFGATHER")' in text
    assert '256: ("wx", 0, 0, "-DNCW=16 -DSTREAM_FINALIZE")' in text
    assert 'M__extX(("-DNCW=16", "-DSTREAM_FINALIZE"))' in text
    assert '"sfX": torch.zeros(5635, dtype=torch.int32, device=dev)' in text
    assert "SF_WORDS (SF_CHUNKS_DONE + 1)" in source
    assert "pass_done == NRB2 * NCW" in source
    assert "done == ldacq(p.sf + SF_TOKEN_EXPECTED + token)" in source
    assert "while (ldacq(p.sf + SF_CHUNKS_DONE) < p.M * 16)" in source
    assert "atomicCAS(p.sf + SF_HEAD, head, head + 1) == head" in source
    assert "mbarrier.try_wait.parity.shared::cta.b64" in source

    # The streamed chunk uses the exact route order, normalization expression,
    # BF16 conversion, and output vector width of the old post-kernel sweep.
    for fragment in (
        "for (int k = 0; k < 8; k++)",
        "p.tw[r] * rsqrtf(__ldcg(p.ss + r) * (1.f / HD) + 1e-8f)",
        "__floats2bfloat162_rn(a[2 * j], a[2 * j + 1])",
        "*(uint4*)(p.out + (size_t)token * HD + col) = o",
    ):
        assert fragment in source


def simulate(ids: list[int], route_start: int, seed: int) -> None:
    expected = [0] * M
    routes_by_expert: dict[int, list[int]] = defaultdict(list)
    for route, expert_id in enumerate(ids):
        expert = expert_id - route_start
        if 0 <= expert < NLOC:
            expected[route // TOPK] += 1
            routes_by_expert[expert].append(route)

    token_done = [0] * M
    enqueued: list[int] = [token for token, count in enumerate(expected) if count == 0]
    pass_done: Counter[tuple[int, int]] = Counter()
    events: list[tuple[int, int]] = []
    for expert, routes in routes_by_expert.items():
        for pass_index in range((len(routes) + TPX - 1) // TPX):
            events.extend([(expert, pass_index)] * PASS_TARGET)
    random.Random(seed).shuffle(events)

    for key in events:
        pass_done[key] += 1
        if pass_done[key] != PASS_TARGET:
            continue
        expert, pass_index = key
        begin = pass_index * TPX
        for route in routes_by_expert[expert][begin : begin + TPX]:
            token = route // TOPK
            token_done[token] += 1
            if token_done[token] == expected[token]:
                enqueued.append(token)

    assert token_done == expected
    assert sorted(enqueued) == list(range(M))
    assert len(enqueued) == len(set(enqueued)) == M
    tasks = [token * TASKS_PER_TOKEN + chunk for token in enqueued for chunk in range(TASKS_PER_TOKEN)]
    assert len(tasks) == M * TASKS_PER_TOKEN
    assert sorted(tasks) == list(range(M * TASKS_PER_TOKEN))
    assert all(count == PASS_TARGET for count in pass_done.values())


def check_event_model() -> None:
    route_start = 100
    simulate([route_start - 1] * (M * TOPK), route_start, 1)
    simulate([route_start] * (M * TOPK), route_start, 2)
    for seed in range(12):
        rng = random.Random(seed)
        ids = [rng.randrange(route_start - 12, route_start + NLOC + 12) for _ in range(M * TOPK)]
        simulate(ids, route_start, seed + 100)


def check_numerical_partition() -> None:
    import torch

    rng = random.Random(19)
    torch.manual_seed(19)
    tokens, hidden, chunk = 23, 512, 32
    route_start = 100
    ids = torch.tensor(
        [rng.randrange(route_start - 4, route_start + NLOC + 4) for _ in range(tokens * TOPK)],
        dtype=torch.int32,
    )
    weights = torch.randn(tokens * TOPK, dtype=torch.float32)
    sums = torch.rand(tokens * TOPK, dtype=torch.float32) * hidden + 1.0
    y = torch.randn(tokens * TOPK, hidden, dtype=torch.float32).to(torch.bfloat16)

    def finish_task(token: int, part: int) -> torch.Tensor:
        lo, hi = part * chunk, (part + 1) * chunk
        acc = torch.zeros(chunk, dtype=torch.float32)
        for k in range(TOPK):
            route = token * TOPK + k
            expert = int(ids[route]) - route_start
            if 0 <= expert < NLOC:
                cf = weights[route] * torch.rsqrt(sums[route] / hidden + 1.0e-8)
                acc += cf * y[route, lo:hi].float()
        return acc.to(torch.bfloat16)

    baseline = torch.empty(tokens, hidden, dtype=torch.bfloat16)
    for token in range(tokens):
        for part in range(hidden // chunk):
            baseline[token, part * chunk : (part + 1) * chunk] = finish_task(token, part)

    streamed = torch.empty_like(baseline)
    tasks = [(token, part) for token in range(tokens) for part in range(hidden // chunk)]
    rng.shuffle(tasks)
    for token, part in tasks:
        streamed[token, part * chunk : (part + 1) * chunk] = finish_task(token, part)
    assert torch.equal(streamed, baseline)


if __name__ == "__main__":
    check_structure()
    check_event_model()
    check_numerical_partition()
    print("S019 streamed-finalization checks passed")
