with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

# Find the main kernel entry points
idx = text.find('def kernel_fn(')
if idx != -1:
    print(text[idx:idx+2500])
else:
    print("kernel_fn not found")
