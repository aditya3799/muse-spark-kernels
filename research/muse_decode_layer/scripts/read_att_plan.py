with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

idx = text.find('def _att_plan')
if idx == -1:
    idx = text.find('att_plan')
if idx == -1:
    idx = text.find('def _launch7')
print(text[idx:idx+1500])
