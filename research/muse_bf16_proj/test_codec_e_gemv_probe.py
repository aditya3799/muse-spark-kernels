import math

import pytest
import torch

import codec_e_gemv_probe as probe


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_codec_e_qkv_m1_matches_fp64_and_one_hot():
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(5151)
    # Four row tiles keep the local test quick while exercising all eight K warps.
    weights = (torch.randn((64, 8192), generator=generator, device=device) / math.sqrt(8192)).to(torch.bfloat16)
    state = probe.prepare_codec_e_gemv(weights)
    assert state["bits_per_weight"] < 12.0
    assert probe.verify_codec_e_gemv_pack(state)
    for stages in (1, 2):
        resources = probe.codec_e_gemv_resource_info(stages)
        assert resources["local_bytes"] == 0
        x = torch.randn((1, 8192), generator=generator, device=device).to(torch.bfloat16)
        output = probe.run_codec_e_gemv(state, x, stages=stages)
        reference = (x.double() @ weights.double().t()).to(torch.bfloat16)
        torch.cuda.synchronize()
        relative = (output.double() - reference.double()).norm() / reference.double().norm()
        assert float(relative) < 5e-3
        for column in (0, 2731, 5461, 8191):
            one = torch.zeros_like(x)
            one[0, column] = 1
            got = probe.run_codec_e_gemv(state, one, stages=stages)
            assert torch.equal(got[0], weights[:, column])
