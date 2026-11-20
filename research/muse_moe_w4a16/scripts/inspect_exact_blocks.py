with open("submission.py") as f:
    text = f.read()

idx_main = text.find("ALT_SRCS['v167_main'] = '")
idx_defines = text.find("V167_MAIN_DEFINES", idx_main)
v167_main = text[idx_main:idx_defines]

def show_block(pattern, name):
    pos = v167_main.find(pattern)
    print(f"=== {name} (found: {pos != -1}) ===")
    if pos != -1:
        print(v167_main[pos:pos+400])

show_block("template <> struct WG<3>", "WG<3>")
show_block("template <> struct WG<4>", "WG<4>")
show_block("template <> struct WGSMMA<3>", "WGSMMA<3>")
show_block("template <> struct WGSMMA<4>", "WGSMMA<4>")
show_block("case 3: fat_compute_wg<3>", "fat_compute_wg switch")
show_block("case 3: fat_compute<3>", "fat_compute switch")
show_block("case 3: wgs_seg<PB, 3>", "wgs_seg switch")
