with open('submission.py') as f:
    text = f.read()

start = text.find('void moe_fused(')
if start == -1:
    start = text.find('okc::moe_fused')
print(f"Start pos: {start}")
print(text[start-200:start+1200])
