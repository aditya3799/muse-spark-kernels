with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

idx = text.find('mbar_init')
while idx != -1:
    print(text[idx-50:idx+200])
    print("-"*50)
    idx = text.find('mbar_init', idx+1)
