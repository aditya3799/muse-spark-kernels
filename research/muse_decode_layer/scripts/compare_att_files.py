with open('/mnt/d/OpenKernelChallenge/muse_attention_decode/submission.py') as f:
    att_top = f.read()

with open('submission.py') as f:
    dec = f.read()

# Extract ATT7_CUDA_SRC from dec
start = dec.find('ATT7_CUDA_SRC = r"""')
end = dec.find('"""\nATT7_CPP_SRC', start)
dec_att = dec[start:end]

print("Standalone attention length:", len(att_top))
print("Embedded attention length:", len(dec_att))

# Find key macro definitions in both
import re
print("\n--- ATTENTION MACRO COMPARISON ---")
for macro in ['OKC_WS', 'OKC_WS8', 'OKC_NP', 'OKC_NP8', 'OKC_ST1F', 'OKC_ST8F', 'OKC_STN', 'OKC_PREGS', 'OKC_TMAPG', 'OKC_F2', 'OKC_S2', 'OKC_CONT', 'OKC_QSS']:
    m1 = re.search(r'#define\s+' + macro + r'\s+([^\n/]+)', att_top)
    m2 = re.search(r'#define\s+' + macro + r'\s+([^\n/]+)', dec_att)
    v1 = m1.group(1).strip() if m1 else 'None'
    v2 = m2.group(1).strip() if m2 else 'None'
    print(f"{macro:<15} Standalone={v1:<10} Decoder={v2:<10} {'DIFF' if v1 != v2 else 'SAME'}")
