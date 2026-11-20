with open("submission.py") as f:
    text = f.read()

# 1. Guard WG<3> and WG<4> in v167_main
t_wg3 = "template <> struct WG<3> {"
t_wg4 = "template <> struct WG<4> {"
t_wg5 = "#if (OKC_FT_T / 8) >= 5\ntemplate <> struct WG<5> {"

# Let's inspect how WG<3> and WG<4> are laid out in v167_main
# They appear inside ALT_SRCS['v167_main']
idx_main = text.find("ALT_SRCS['v167_main'] = '")
idx_defines = text.find("V167_MAIN_DEFINES", idx_main)
v167_main = text[idx_main:idx_defines]

print("Found v167_main, length:", len(v167_main))
print("t_wg3 in v167_main?", t_wg3 in v167_main)
print("WGSMMA<3> in v167_main?", "template <> struct WGSMMA<3> {" in v167_main)
