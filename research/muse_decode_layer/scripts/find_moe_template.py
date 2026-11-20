with open('submission.py') as f:
    text = f.read()

idx = text.find('__global__ void moe_fused')
if idx == -1:
    idx = text.find('void moe_fused(')
if idx == -1:
    idx = text.find('moe_fused(')
print(f"Pos: {idx}")
print(text[idx-200:idx+800])
