with open('submission.py', 'r', encoding='utf-8', errors='ignore') as f:
    text_sub = f.read()

idx1 = text_sub.find('def _make_chen():')
idx2 = text_sub.find('\n# ---------------------------------------------------------------------------------------------- rs128 (chenwainuo)', idx1)
sub_chen = text_sub[idx1:idx2]

lines = sub_chen.splitlines()
for i in range(len(lines) - 60, len(lines)):
    print(f"{i}: {lines[i]}")
