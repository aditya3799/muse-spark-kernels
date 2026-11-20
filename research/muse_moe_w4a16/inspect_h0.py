with open("submission.py", "r", encoding="utf-8") as f:
    text = f.read()

# Find where ALT_SRCS['v167_helper0'] is defined and print summary
pos = text.find("ALT_SRCS['v167_helper0']")
end = text.find("ALT_SRCS['v167_main']")
h0_src = text[pos:end]
print(f"v167_helper0 length: {len(h0_src)} chars")

# Check for key features in v167_helper0
for term in ["moe_fused", "skinny_ring", "okc_run", "CTA_SYNC", "WS_JOIN", "ld_acquire", "doneA", "WSH0"]:
    print(f"'{term}' in v167_helper0: {term in h0_src}")
