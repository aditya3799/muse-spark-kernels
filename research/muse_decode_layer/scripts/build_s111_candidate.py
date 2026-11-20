with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update module name to s111b
old_mod = 'muse_att_v298w1pf_s108_{arch}'
assert old_mod in text, f"Module {old_mod} not found in submission.py"
text = text.replace(old_mod, 'muse_att_v298w1pf_s111b_{arch}')

# 2. Add CONT to struct Cfg
old_cfg = '''  // flat kernel on sm_90: one extra producer warp issues every K/V copy (warp specialization), so the
  // consumers never stall in cp.async issue; the stages are handed over with mbarriers
  static constexpr bool WS = F && OKC_SM90 && OKC_WS && (L == 1 || OKC_WS8);'''
assert old_cfg in text, "old_cfg not found"
new_cfg = '''  // flat kernel on sm_90: one extra producer warp issues every K/V copy (warp specialization), so the
  // consumers never stall in cp.async issue; the stages are handed over with mbarriers
  static constexpr bool WS = F && OKC_SM90 && OKC_WS && (L == 1 || OKC_WS8);
  static constexpr bool CONT = F && (L == 1) && WS;'''
text = text.replace(old_cfg, new_cfg, 1)

# 3. Replace constexpr bool CONT in segment
old_cont_def = 'constexpr bool CONT = FLAT && Cfg<L, FLAT>::WS && OKC_CONT;'
assert old_cont_def in text, "old_cont_def not found"
new_cont_def = 'constexpr bool CONT = Cfg<L, FLAT>::CONT;'
text = text.replace(old_cont_def, new_cont_def, 1)

# 4. Add constexpr bool CONT in attn_kernel head
old_attn_head = '''__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel(const __grid_constant__ Params p) {
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;'''
assert old_attn_head in text, "old_attn_head not found"
new_attn_head = '''__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel(const __grid_constant__ Params p) {
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  constexpr bool CONT = Cfg<L, FLAT>::CONT;'''
text = text.replace(old_attn_head, new_attn_head, 1)

# 5. In attn_kernel: if constexpr (Cfg<L, FLAT>::WS && OKC_CONT)
old_cont_bar = 'if constexpr (Cfg<L, FLAT>::WS && OKC_CONT) {'
assert old_cont_bar in text, "old_cont_bar not found"
new_cont_bar = 'if constexpr (Cfg<L, FLAT>::WS && CONT) {'
text = text.replace(old_cont_bar, new_cont_bar, 1)

# 6. In segment call: OKC_CONT ? tb : 0
old_seg_tb = 'OKC_CONT ? tb : 0, 0,'
assert old_seg_tb in text, "old_seg_tb not found"
new_seg_tb = 'CONT ? tb : 0, 0,'
text = text.replace(old_seg_tb, new_seg_tb, 1)

# 7. In segment call earlybar condition: !OKC_CONT
old_earlybar = 'OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !OKC_CONT && !OKC_PTR && k == 0'
assert old_earlybar in text, "old_earlybar not found"
new_earlybar = 'OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !CONT && !OKC_PTR && k == 0'
text = text.replace(old_earlybar, new_earlybar, 1)

# 8. In producer exit: if constexpr (R == 1 && OKC_CONT) cp_wait<0>();
old_wait = 'if constexpr (R == 1 && OKC_CONT) cp_wait<0>();'
assert old_wait in text, "old_wait not found"
new_wait = 'if constexpr (R == 1 && CONT) cp_wait<0>();'
text = text.replace(old_wait, new_wait, 1)

out_file = 'submission_s111_cont_l1_only.py'
with open(out_file, 'w', encoding='utf-8', newline='\n') as f:
    f.write(text)

import os
size = os.path.getsize(out_file)
print(f"Successfully generated {out_file}, size = {size} bytes")
assert size < 1000000, f"Size {size} exceeds 1,000,000 bytes!"
