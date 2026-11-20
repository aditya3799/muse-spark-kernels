import re

with open("submission.py") as f:
    orig = f.read()

# Let's locate ALT_SRCS['v167_main']
idx = orig.find("ALT_SRCS['v167_main'] = '")
if idx == -1:
    print("Not found")
    exit(1)

# Find the end of this string assignment
# The string assignment is `ALT_SRCS['v167_main'] = '...'\n`
end_idx = orig.find("\n\n'\nV167_MAIN_DEFINES", idx)
print(f"Start: {idx}, End: {end_idx}")

# Let's extract the string
src_quoted = orig[idx + len("ALT_SRCS['v167_main'] = '") : end_idx + 3]
print(f"Length of v167_main raw string: {len(src_quoted)}")
