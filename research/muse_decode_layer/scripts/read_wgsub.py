with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

idx = text.find('OKC_WGSUB')
print(text[idx-200:idx+300])
