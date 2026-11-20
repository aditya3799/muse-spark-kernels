import math

import torch

import codec_e_gemv_probe as probe
from codec_e import unpack_warp_interleaved


device = "cuda"
generator = torch.Generator(device=device).manual_seed(5151)
weights = (torch.randn((16, 8192), generator=generator, device=device) / math.sqrt(8192)).to(torch.bfloat16)
state = probe.prepare_codec_e_gemv(weights)
print("bits", state["bits_per_weight"])
print("stream_exact", bool(torch.equal(unpack_warp_interleaved(state["_debug_packed"]), state["_debug_streams"])))
recovered, original = probe.recover_codec_e_gemv_bits(state)
print("coefficient_shapes", tuple(recovered.shape), tuple(original.shape))
mismatch = (recovered != original).nonzero()
print("coefficient_mismatches", mismatch.shape[0])
if mismatch.numel():
    index = tuple(mismatch[0].tolist())
    print("first_coefficient_mismatch", index, int(recovered[index]), int(original[index]))
print("resources", probe.codec_e_gemv_resource_info(1), probe.codec_e_gemv_resource_info(2))
for column in (0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 1023, 8191):
    x = torch.zeros((1, 8192), dtype=torch.bfloat16, device=device)
    x[0, column] = 1
    got = probe.run_codec_e_gemv(state, x, stages=1)[0]
    expected = weights[:, column]
    torch.cuda.synchronize()
    mismatch = (got != expected).nonzero().flatten()
    print("column", column, "mismatches", int(mismatch.numel()),
          "got", got.float().cpu().tolist(), "expected", expected.float().cpu().tolist())
