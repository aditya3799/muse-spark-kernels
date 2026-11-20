with open('submission.py') as f:
    text = f.read()

idx = text.find('state["choice"] = {}')
print(text[idx-500:idx+1500])
