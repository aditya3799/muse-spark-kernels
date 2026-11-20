with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

idx = text.find('def wgs_prefix')
if idx == -1:
    idx = text.find('wgs_prefix')
print(text[idx:idx+1500])
