"""Generate v277: unrolled 1/2/4 row grouping with predecoded deltas."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pairplanes_rowgroup_v272.py"
OUTPUT = ROOT / "submission_plain_pairplanes_group_predelta_unrolled_v277.py"

text = BASE.read_text(encoding="utf-8")
text = text.replace("v272", "v277").replace("V272", "V277")

old = "esc_delta((uint32_t)en.y, base4 & 0xFFu)"
if text.count(old) < 1:
    raise RuntimeError("v277 escape decode marker mismatch")
pos = text.rfind(old)
text = text[:pos] + "__uint_as_float((uint32_t)en.y)" + text[pos + len(old):]

marker = "\ndef _v277_run(x, layer, n, state, group):\n"
if text.count(marker) != 1:
    raise RuntimeError("v277 run marker mismatch")
delta_fn = r'''

def _v277_delta_table(cg):
    """Replace stored BF16 escape bits by identical FP32 delta bits."""
    tab = cg['tab'].clone()
    bits = tab[:, :, 1]
    base = int(cg['base4']) & 0xFF
    hi = (bits >> 8) & 0xFF
    e71 = hi & 0x7F
    dhi = (hi & 0x80) | torch.where(
        e71 < base, torch.full_like(e71, base),
        torch.full_like(e71, base + 7))
    dbits = ((dhi << 8) | (bits & 0xFF)) & 0xFFFF
    original = ((bits & 0xFFFF).to(torch.int16).contiguous()
                .view(torch.bfloat16).float())
    decoded = (dbits.to(torch.int16).contiguous()
               .view(torch.bfloat16).float())
    delta = (original - decoded).contiguous()
    count = tab[:, 0, 0]
    slot = torch.arange(tab.shape[1], device=tab.device).view(1, -1)
    used = (slot >= 1) & (slot <= count.view(-1, 1))
    exact = bool(torch.equal((decoded + delta)[used], original[used]))
    tab[:, :, 1] = delta.view(torch.int32)
    return tab.contiguous(), exact
'''
text = text.replace(marker, delta_fn + marker, 1)

old = """    _get_ext().cgemv_pair_group(entry['packed'], x, cg['tab'], cg['pmax'],
                                out, cg['base4'], entry['nth'], group)"""
new = """    _get_ext().cgemv_pair_group(entry['packed'], x, entry['table'],
                                cg['pmax'], out, cg['base4'],
                                entry['nth'], group)"""
if text.count(old) != 1:
    raise RuntimeError("v277 launch table marker mismatch")
text = text.replace(old, new, 1)

old = """            pair, pack_exact = _v106_pair_pack(state['cg'][layer])
            entry = {'packed': pair, 'nth': nth, 'enabled': False,
                     'group': 0, 'pack_exact': pack_exact}"""
new = """            pair, pack_exact = _v106_pair_pack(state['cg'][layer])
            table, table_exact = _v277_delta_table(state['cg'][layer])
            entry = {'packed': pair, 'table': table, 'nth': nth,
                     'enabled': False, 'group': 0,
                     'pack_exact': pack_exact, 'table_exact': table_exact}"""
if text.count(old) != 1:
    raise RuntimeError("v277 prepare table marker mismatch")
text = text.replace(old, new, 1)

old = """                valid = (pack_exact and bitwise and probe_exact and finite and
                         rel < 5e-3 and resources['local_bytes'] == 0 and"""
new = """                valid = (pack_exact and table_exact and bitwise and
                         probe_exact and finite and rel < 5e-3 and
                         resources['local_bytes'] == 0 and"""
if text.count(old) != 1:
    raise RuntimeError("v277 validity marker mismatch")
text = text.replace(old, new, 1)

old = """                      'resources', resources, 'pack_exact', pack_exact,
                      'bitwise_base', bitwise, 'probe_exact', probe_exact,"""
new = """                      'resources', resources, 'pack_exact', pack_exact,
                      'table_exact', table_exact,
                      'bitwise_base', bitwise, 'probe_exact', probe_exact,"""
if text.count(old) != 1:
    raise RuntimeError("v277 print marker mismatch")
text = text.replace(old, new, 1)

compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
