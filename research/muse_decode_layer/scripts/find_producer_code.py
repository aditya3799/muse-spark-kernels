with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

idx = text.find('// producer')
while idx != -1:
    print("="*60)
    print(text[idx:idx+400])
    idx = text.find('// producer', idx+1)
