"""Exhaustively validate the packed-half WST polynomial over all FP16 inputs."""

import torch


bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
x = bits.view(torch.float16)
finite_or_neginf = (~torch.isnan(x)) & (x <= torch.tensor(8.75, dtype=torch.float16))
x = x[finite_or_neginf]


def h(v):
    return v.to(torch.float16)


# Mirror the CUDA half2 dataflow: each HFMA has a single FP32 evaluation followed by FP16 rounding.
xc = torch.maximum(x, torch.tensor(-14.0, dtype=torch.float16))
j = torch.floor(xc).to(torch.float16)
f = h(xc.float() - j.float())
c0, c1 = h(torch.tensor(0.0790209)), h(torch.tensor(0.2249424))
c2, c3 = h(torch.tensor(0.6960656)), h(torch.tensor(0.9999893))
p = h(f.float() * c0.float() + c1.float())
p = h(f.float() * p.float() + c2.float())
p = h(f.float() * p.float() + c3.float())

pb = p.view(torch.uint16).to(torch.int32)
approx_bits = ((pb + j.to(torch.int32) * 1024) & 0xFFFF).to(torch.uint16)
approx = approx_bits.view(torch.float16)
exact = torch.exp2(x.float()).to(torch.float16)

rel = (approx.float() - exact.float()).abs() / exact.float().clamp_min(2.0 ** -24)
print("inputs", x.numel())
print("max_relative_f16", rel.max().item())
print("mean_relative_f16", rel.mean().item())
relevant = x >= torch.tensor(-10.0, dtype=torch.float16)
print("max_relative_f16_e4m3_nonzero_domain", rel[relevant].max().item())
print("mean_relative_f16_e4m3_nonzero_domain", rel[relevant].mean().item())
print("nonfinite_approx", (~torch.isfinite(approx)).sum().item())

# The judge consumes E4M3 P, so this is the decision-level comparison that matters.
qa = approx.float().to(torch.float8_e4m3fn).view(torch.uint8)
qe = torch.exp2(x.float()).to(torch.float8_e4m3fn).view(torch.uint8)
diff = qa != qe
print("e4m3_code_mismatches", diff.sum().item())
if diff.any():
    xd = x[diff].float()
    print("mismatch_x_range", xd.min().item(), xd.max().item())
    print("max_code_delta", (qa[diff].to(torch.int16) - qe[diff].to(torch.int16)).abs().max().item())
