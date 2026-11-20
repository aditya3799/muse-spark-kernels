"""CPU round-trip test for the v278 C13 row-pair byte layout."""

import runpy
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
ns = runpy.run_path(str(ROOT / "submission_plain_pairplanes_c13_nohelper_v278.py"))
pack = ns["_v278_pack"]

g = torch.Generator().manual_seed(27813)
w = torch.randn((6, 64), generator=g).to(torch.bfloat16)
first = pack(w)
base = first["base4"] & 0xFF
bits = (w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF)
bits[0, 0] = ((max(base - 1, 0) << 8) | 0x80)
bits[-1, -1] = ((min(base + 16, 127) << 8) | 0x40)
w = bits.to(torch.int16).contiguous().view(torch.bfloat16)
d = pack(w)

n, k = w.shape
chunks = k // 16
p = d["packed"]
low0 = p[:, :16 * chunks].view(n // 2, chunks, 16)
low1 = p[:, 16 * chunks:32 * chunks].view(n // 2, chunks, 16)
low = torch.stack((low0, low1), 1).reshape(n, k).to(torch.int32)
code_pair = p[:, 32 * chunks:48 * chunks].view(n // 2, chunks, 16)
codes = torch.stack((code_pair[:, :, :8], code_pair[:, :, 8:]), 1)
codes = codes.reshape(n, chunks, 2, 4)
offset = torch.cat((codes & 0xF, codes >> 4), -1).reshape(n, k).to(torch.int32)
sign_pair = p[:, 48 * chunks:].view(n // 2, chunks, 4)
sign_bytes = torch.stack((sign_pair[:, :, :2], sign_pair[:, :, 2:]), 1)
sign_bytes = sign_bytes.reshape(n, chunks, 2).to(torch.int32)
sign16 = sign_bytes[:, :, 0] | (sign_bytes[:, :, 1] << 8)
shifts = torch.arange(16, dtype=torch.int32)
sign = ((sign16.unsqueeze(-1) >> shifts) & 1).reshape(n, k)
decoded_bits = ((((sign << 7) | ((d["base4"] & 0xFF) + offset)) << 8) |
                low).to(torch.int16)
restored = decoded_bits.contiguous().view(torch.bfloat16).float()
table = d["table"]
for cta in range(table.shape[0]):
    for slot in range(1, int(table[cta, 0, 0]) + 1):
        loc = int(table[cta, slot, 0])
        row = 2 * cta + (loc >> 16)
        col = loc & 0xFFFF
        delta = table[cta, slot, 1].reshape(1).view(torch.float32)[0]
        restored[row, col] += delta

assert d["pack_exact"] and d["delta_exact"]
assert torch.equal(restored, w.float())
assert d["escapes"] >= 1
print("V278 C13 FULL CPU ROUND TRIP OK", d["escapes"], p.numel())
