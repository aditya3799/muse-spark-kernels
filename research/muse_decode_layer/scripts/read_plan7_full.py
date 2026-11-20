with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

pos = text.find('def _plan7(')
print(text[pos:pos+1500])
