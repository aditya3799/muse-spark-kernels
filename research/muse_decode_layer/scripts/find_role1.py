with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

pos = text.find('if constexpr (ROLE == 1)')
while pos != -1:
    print("="*60)
    print(text[pos:pos+1500])
    pos = text.find('if constexpr (ROLE == 1)', pos+1)
