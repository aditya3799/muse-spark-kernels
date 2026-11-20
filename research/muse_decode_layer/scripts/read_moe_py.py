with open('submission.py') as f:
    text = f.read()

pos = text.find('def _moe(')
print(text[pos:pos+2500])
