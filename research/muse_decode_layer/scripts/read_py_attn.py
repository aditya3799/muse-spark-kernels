with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

pos = text.find('(_mod or _att7()).run(')
line_start = text.rfind('\n', 0, pos - 500)
line_end = text.find('\n', pos + 800)
print(text[line_start:line_end])
