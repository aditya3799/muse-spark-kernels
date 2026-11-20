with open('submission_s114_rq_qkv_fixed2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. In struct Cfg: add CONT definition
old_cfg = "  static constexpr bool WS = F && OKC_SM90 && OKC_WS && (L == 1 || OKC_WS8);"
new_cfg = "  static constexpr bool WS = F && OKC_SM90 && OKC_WS && (L == 1 || OKC_WS8);\n  static constexpr bool CONT = F && (L == 1) && WS;"
assert content.count(old_cfg) == 1, "old_cfg anchor not unique"
content = content.replace(old_cfg, new_cfg)

# 2. In segment(): replace OKC_CONT with Cfg<L, FLAT>::CONT
old_seg_cont = "  constexpr bool CONT = FLAT && Cfg<L, FLAT>::WS && OKC_CONT;"
new_seg_cont = "  constexpr bool CONT = Cfg<L, FLAT>::CONT;"
assert content.count(old_seg_cont) == 1, "old_seg_cont anchor not unique"
content = content.replace(old_seg_cont, new_seg_cont)

# 3. In attn_kernel(): define CONT at top of function (uniquely anchored)
old_attn_top = """__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel(const __grid_constant__ Params p) {
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;"""
new_attn_top = """__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel(const __grid_constant__ Params p) {
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  constexpr bool CONT = Cfg<L, FLAT>::CONT;"""
assert content.count(old_attn_top) == 1, "old_attn_top anchor not unique"
content = content.replace(old_attn_top, new_attn_top)

# 4. In attn_kernel(): replace OKC_CONT check with CONT
old_attn_cont = "  if constexpr (Cfg<L, FLAT>::WS && OKC_CONT) {   // continuous pipeline: stage barriers set up once per launch"
new_attn_cont = "  if constexpr (Cfg<L, FLAT>::WS && CONT) {   // continuous pipeline: stage barriers set up once per launch"
assert content.count(old_attn_cont) == 1, "old_attn_cont anchor not unique"
content = content.replace(old_attn_cont, new_attn_cont)

# 5. In segment() load_kv_stage call: replace OKC_CONT with CONT
old_load_kv = """                            OKC_CONT ? tb : 0, 0,
                            OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !OKC_CONT && !OKC_PTR && k == 0);"""
new_load_kv = """                            CONT ? tb : 0, 0,
                            OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !CONT && !OKC_PTR && k == 0);"""
assert content.count(old_load_kv) == 1, "old_load_kv anchor not unique"
content = content.replace(old_load_kv, new_load_kv)

# 6. In segment() producer exit: replace OKC_CONT with CONT
old_prod_exit = "    if constexpr (R == 1 && OKC_CONT) cp_wait<0>();   // the producer leaves with its copies landed"
new_prod_exit = "    if constexpr (R == 1 && CONT) cp_wait<0>();   // the producer leaves with its copies landed"
assert content.count(old_prod_exit) == 1, "old_prod_exit anchor not unique"
content = content.replace(old_prod_exit, new_prod_exit)

# 7. Update module name for inline compilation
content = content.replace('muse_att_v298w1pf_s114_', 'muse_att_v298w1pf_s118_')

# Convert strictly to Unix LF (\n)
content_bytes = content.encode('utf-8').replace(b'\r\n', b'\n')
print("Size of s118:", len(content_bytes), "bytes")
assert len(content_bytes) < 1000000, "File too large!"

with open('submission_s118_cont_l1_rq_qkv.py', 'wb') as f:
    f.write(content_bytes)

print("Created submission_s118_cont_l1_rq_qkv.py successfully!")
