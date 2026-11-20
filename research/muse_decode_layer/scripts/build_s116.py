with open('submission_s114_rq_qkv_fixed2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Verify targets exist
assert '#define OKC_WGSU 160' in content
assert '#define OKC_WGSUB 140' in content

# Replace
content = content.replace('#define OKC_WGSU 160', '#define OKC_WGSU 80')
content = content.replace('#define OKC_WGSUB 140', '#define OKC_WGSUB 70')

# Update build name so cached ninja / torch inline builds do not collide
content = content.replace('muse_att_v298w1pf_s114_', 'muse_att_v298w1pf_s116_')
content = content.replace('okc_helper_m16_s108_', 'okc_helper_m16_s116_')

print("Size of s116:", len(content.encode('utf-8')), "bytes")
assert len(content.encode('utf-8')) < 1000000

with open('submission_s116_cepi_u80.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Created submission_s116_cepi_u80.py successfully!")
