with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

idx = text.find('segment<')
while idx != -1:
    print(text[idx-50:idx+150])
    print("-"*50)
    idx = text.find('segment<', idx+1)
