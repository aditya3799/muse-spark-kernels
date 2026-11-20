"""Generate v275: v274 plus a uniform no-escape CTA fast path."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pairplanes_predelta_v274.py"
OUTPUT = ROOT / "submission_plain_pairplanes_predelta_zero_v275.py"

text = BASE.read_text(encoding="utf-8")
text = text.replace("v274", "v275").replace("V274", "V275")

old = """    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS] = {0.f, 0.f};
    for (int i0 = 0; i0 <= cnt; i0 += 32) {"""
new = """    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    if (cnt == 0) {
      if (lane == 0) {
        red[0][NWARP] = 0.f;
        red[1][NWARP] = 0.f;
      }
      __syncthreads();
      return;
    }
    float ec[ROWS] = {0.f, 0.f};
    for (int i0 = 0; i0 <= cnt; i0 += 32) {"""
if text.count(old) < 1:
    raise RuntimeError("v275 helper fast-path marker mismatch")
pos = text.rfind(old)
text = text[:pos] + new + text[pos + len(old):]

old = """            resources = _v275_resource_info(k, nth)
            gen = torch.Generator(device=wq.device).manual_seed(274101 + li)"""
new = """            resources = _v275_resource_info(k, nth)
            counts = state['cg'][layer]['tab'][:, 0, 0]
            zero_ctas = int((counts == 0).sum().item())
            total_ctas = int(counts.numel())
            print('v275 pair predelta zero histogram', layer,
                  'zero_ctas', zero_ctas, 'total_ctas', total_ctas,
                  'zero_fraction', zero_ctas / max(total_ctas, 1), flush=True)
            gen = torch.Generator(device=wq.device).manual_seed(275101 + li)"""
if text.count(old) != 1:
    raise RuntimeError("v275 histogram marker mismatch")
text = text.replace(old, new, 1)

compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
