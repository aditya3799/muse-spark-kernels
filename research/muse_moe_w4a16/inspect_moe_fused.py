with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find("ALT_SRCS['v167_main'] = '")
end = text.find("'\n", idx)
src = text[idx:end]
lines = src.split('\\n')
for i, l in enumerate(lines):
    if '__global__ void __launch_bounds__(NTL, 1) moe_fused' in l:
        print('\\n'.join(lines[i:i+80]))
        break
