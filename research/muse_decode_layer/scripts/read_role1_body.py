with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

pos = text.find('if constexpr (ROLE == 1)')
print(text[pos:pos+3500])
