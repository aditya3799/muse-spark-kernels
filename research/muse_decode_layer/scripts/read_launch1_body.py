with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

idx = text.find('void launch(const Params& p0')
print(text[idx:idx+2500])
