with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

chen_idx = text.find('def _make_chen():')
chen_code = text[chen_idx:chen_idx+20000]
print("griddepcontrol in chen:", "griddepcontrol" in chen_code)
print("pdl in chen:", "pdl" in chen_code.lower())
