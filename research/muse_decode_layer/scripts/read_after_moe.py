with open('submission.py') as f:
    text = f.read()

idx = text.find('moe = _pick_expert_backend')
print(text[idx:idx+2500])
