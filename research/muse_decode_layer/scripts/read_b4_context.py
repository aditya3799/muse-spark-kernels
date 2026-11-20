with open('submission_s109_cont1.py') as f:
    text = f.read()
idx = text.find('key, defines = "b4"')
print(text[idx-300:idx+400])
