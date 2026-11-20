"""Local CUDA audit for S018's exact in-place residual/RMS boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "submission_s018_prompt_boundaries.py"


def load_candidate_functions():
    source = CANDIDATE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"_s018_reuse_boundary_kernel", "_s018_reuse_boundary"}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted]
    assert {node.name for node in nodes} == wanted
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {"torch": torch, "triton": triton, "tl": tl}
    exec(compile(module, str(CANDIDATE), "exec"), namespace)
    return namespace["_s018_reuse_boundary"]


def reference(branch, stream, post_weight, pre_weight, alpha, beta, post_input_scale):
    post_input = branch.float()
    if post_input_scale:
        post_input = (post_input * post_weight.float()).to(torch.bfloat16).float()
    normed_branch = post_input * torch.rsqrt((post_input * post_input).mean(1, keepdim=True) + 1.0e-8)
    normed_branch = normed_branch.to(torch.bfloat16).float()
    new_stream = alpha * stream + beta * normed_branch
    output = new_stream * torch.rsqrt((new_stream * new_stream).mean(1, keepdim=True) + 1.0e-5)
    output = (output * pre_weight.float()).to(torch.bfloat16)
    return output, new_stream


def make_boundary(device, post_input_scale):
    hidden = 8192
    post_weight = torch.randn(hidden, device=device, dtype=torch.float32).to(torch.bfloat16)
    pre_weight = torch.randn(hidden, device=device, dtype=torch.float32).to(torch.bfloat16)
    alpha = torch.rand(hidden, device=device, dtype=torch.float32)
    beta = torch.rand(hidden, device=device, dtype=torch.float32)
    boundary = SimpleNamespace(
        _cc_post_weight=post_weight if post_input_scale else torch.empty(0, device=device, dtype=torch.bfloat16),
        _cc_pre_weight=pre_weight,
        _cc_alpha=alpha,
        _cc_beta=beta,
        _post_norm=SimpleNamespace(eps=1.0e-8, has_weight=post_input_scale),
        _pre_norm=SimpleNamespace(eps=1.0e-5, has_weight=True),
        _post_norm_is_input_scale=post_input_scale,
    )
    return boundary, post_weight, pre_weight, alpha, beta


def relative_l2(got, expected):
    return float((got.float() - expected.float()).norm() / expected.float().norm())


def graph_audit(reuse_boundary, device):
    """Capture both boundaries and prove the first FP32 workspace is recycled."""
    rows = 8
    attn, _, attn_pre, attn_alpha, attn_beta = make_boundary(device, False)
    ffn, ffn_post, ffn_pre, ffn_alpha, ffn_beta = make_boundary(device, True)
    stream0 = torch.randn((rows, 8192), device=device, dtype=torch.float32)
    attn_branch = torch.randn((rows, 8192), device=device, dtype=torch.float32).to(torch.bfloat16)
    ffn_branch = torch.randn((rows, 8192), device=device, dtype=torch.float32).to(torch.bfloat16)

    # Warm both specializations before capture.
    warm_attn = attn_branch.clone()
    _, warm_stream = reuse_boundary(stream0.clone(), warm_attn, attn, reuse_stream=False)
    warm_ffn = ffn_branch.clone()
    reuse_boundary(warm_stream, warm_ffn, ffn, reuse_stream=True)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attn_out, shared_stream = reuse_boundary(stream0, attn_branch, attn, reuse_stream=False)
        next_input, final_stream = reuse_boundary(shared_stream, ffn_branch, ffn, reuse_stream=True)

    assert attn_out.data_ptr() == attn_branch.data_ptr()
    assert next_input.data_ptr() == ffn_branch.data_ptr()
    assert final_stream.data_ptr() == shared_stream.data_ptr()
    assert final_stream.data_ptr() != stream0.data_ptr()

    for _ in range(3):
        new_stream = torch.randn_like(stream0)
        new_attn = torch.randn_like(attn_branch.float()).to(torch.bfloat16)
        new_ffn = torch.randn_like(ffn_branch.float()).to(torch.bfloat16)
        expected_attn, expected_mid = reference(
            new_attn, new_stream, attn._cc_post_weight, attn_pre, attn_alpha, attn_beta, False)
        expected_next, expected_final = reference(
            new_ffn, expected_mid, ffn_post, ffn_pre, ffn_alpha, ffn_beta, True)
        stream0.copy_(new_stream)
        attn_branch.copy_(new_attn)
        ffn_branch.copy_(new_ffn)
        graph.replay()
        torch.cuda.synchronize()
        assert relative_l2(attn_out, expected_attn) < 0.003
        assert relative_l2(next_input, expected_next) < 0.003
        assert relative_l2(final_stream, expected_final) < 0.003


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(18)
    device = torch.device("cuda")
    reuse_boundary = load_candidate_functions()

    for rows in (1, 8, 32, 128, 256):
        for post_input_scale, reuse_stream in ((False, False), (True, True)):
            branch = torch.randn((rows, 8192), device=device, dtype=torch.float32).to(torch.bfloat16)
            stream = torch.randn((rows, 8192), device=device, dtype=torch.float32)
            boundary, post_weight, pre_weight, alpha, beta = make_boundary(device, post_input_scale)
            expected_output, expected_stream = reference(
                branch, stream, post_weight, pre_weight, alpha, beta, post_input_scale)
            branch_ptr = branch.data_ptr()
            stream_ptr = stream.data_ptr()
            original_stream = stream.clone()
            output, new_stream = reuse_boundary(
                stream, branch, boundary, reuse_stream=reuse_stream)
            torch.cuda.synchronize()

            assert output.data_ptr() == branch_ptr
            assert (new_stream.data_ptr() == stream_ptr) == reuse_stream
            if not reuse_stream:
                assert torch.equal(stream, original_stream)
            assert relative_l2(output, expected_output) < 0.003
            assert relative_l2(new_stream, expected_stream) < 0.003

    graph_audit(reuse_boundary, device)
    print("PASS: S018 CUDA math, alias ownership, input immutability, and graph replay")


if __name__ == "__main__":
    main()
