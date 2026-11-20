with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find("ALT_SRCS['v167_wsi'] = '")
end = text.find("'\n", idx)
src = text[idx:end]

lines = src.split('\\n')
for i, l in enumerate(lines):
    if 'TORCH_CHECK' in l or 'static_assert' in l or 'assert' in l or 'M <=' in l or 'M >' in l or '16' in l and 'p.M' in l:
        print(f"L{i}: {l}")
