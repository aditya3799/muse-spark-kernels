with open('scripts/build_s100_tuned_q1occ3.py') as f:
    text = f.read()

pos = text.find('def _attention_base')
print(text[pos:pos+1500])
