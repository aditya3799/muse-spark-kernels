with open("submission.py") as f:
    text = f.read()

import re
print("TINY_DEFINES:", re.search(r"TINY_DEFINES = ([^\n]+)", text).group(1))
print("TINY2_DEFINES:", re.search(r"TINY2_DEFINES = ([^\n]+)", text).group(1))
print("MID_DEFINES:", re.search(r"MID_DEFINES = ([^\n]+)", text).group(1))
print("V167_M32_DEFINES:", re.search(r"V167_M32_DEFINES = ([^\n]+)", text).group(1))
print("MAIN_DEFINES:", re.search(r"MAIN_DEFINES = ([^\n]+)", text).group(1))
print("TINY_MAX_M:", re.search(r"TINY_MAX_M = ([^\n]+)", text).group(1))
print("TINY2_MAX_M:", re.search(r"TINY2_MAX_M = ([^\n]+)", text).group(1))
print("MID_MAX_M:", re.search(r"MID_MAX_M = ([^\n]+)", text).group(1))
