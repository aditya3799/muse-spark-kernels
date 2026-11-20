with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

# Search for the string containing #define OKC_WGSU
idx = text.find('#define OKC_WGSU')
# Find the surrounding quotes or variable assignment
start = text.rfind('"""', 0, idx)
if start == -1:
    start = text.rfind("'''", 0, idx)
print(f"Start of docstring: {start}")
print(text[start:start+100])

end = text.find('"""', idx)
if end == -1:
    end = text.find("'''", idx)
print(f"End of docstring: {end}")
