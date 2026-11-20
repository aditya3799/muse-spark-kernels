with open('submission_s081_keell0renz_890809c.py') as f:
    s081 = f.read()

with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission_v178_hybrid_m32_retry.py') as f:
    v178 = f.read()

# Find definitions around OKC_WGSU in both
def get_moe_block(text):
    start = text.find('#define OKC_WGSU')
    end = text.find('void moe_fused(', start)
    if end == -1:
        end = text.find('void run(', start)
    return text[start:start+2500] if start != -1 else ""

print("=== S081 MoE segment ===")
print(get_moe_block(s081)[:500])
print("\n=== v178 MoE segment ===")
print(get_moe_block(v178)[:500])
