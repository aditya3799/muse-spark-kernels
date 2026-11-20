with open('submission_s111_cont_l1_only.py') as f:
    text = f.read()

idx = text.find('segment<L, FLAT, R>')
print(text[max(0, idx-400):min(len(text), idx+300)])
