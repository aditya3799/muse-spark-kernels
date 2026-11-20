with open("submission.py") as f:
    text = f.read()

text = text.replace("'-DOKC_FT_T=16'", "'-DOKC_FT_T=32'", 1)
with open("submission.py", "w") as f:
    f.write(text)

print("Reverted to FT_T=32")
