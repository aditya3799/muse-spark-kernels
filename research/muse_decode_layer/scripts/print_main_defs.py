with open('submission.py') as f:
    text = f.read()

pos = text.find('def _main_defines')
print(text[pos:pos+500])
