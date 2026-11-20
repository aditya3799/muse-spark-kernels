import re

with open('profiles/submission_s114_rq_qkv_fixed2.843243d902ab4b5f.txt', 'r', encoding='utf-8') as f:
    text = f.read()

sections = re.split(r'profile \d+:', text)[1:]
names = ['d1_sl', 'd1_fl', 'd16_sl', 'd16_fl', 'd32_sl', 'd32_fl', 'v1_sl', 'v1_fl', 'v16_sl', 'v16_fl', 'v32_sl', 'v32_fl']

print(f"{'Shape':<10} | {'Total':<8} | {'MoE':<8} | {'MoE %':<8}")
print('-' * 40)
for i in range(12):
    sec = sections[i]
    m_moe = re.search(r'okc::moe_fused.*?(\d+\.\d+)\s*(?:µs|us)', sec)
    m_tot = re.search(r'total\s+(\d+\.\d+)\s*(?:µs|us)', sec)
    moe_t = float(m_moe.group(1)) if m_moe else 0.0
    tot_t = float(m_tot.group(1)) if m_tot else 0.0
    pct = (moe_t / tot_t) * 100.0 if tot_t > 0 else 0
    print(f"{names[i]:<10} | {tot_t:<8.1f} | {moe_t:<8.1f} | {pct:<7.1f}%")
