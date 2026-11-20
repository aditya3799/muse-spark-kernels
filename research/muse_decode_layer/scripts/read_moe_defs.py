with open('submission.py') as f:
    text = f.read()

pos = text.find('MOE_TINY_DEFINES =')
print(text[pos:pos+1000])
