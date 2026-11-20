with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

idx = text.find('def _try')
print(text[idx-50:idx+200])
