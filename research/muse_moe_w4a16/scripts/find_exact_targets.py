with open("submission.py") as f:
    orig = f.read()

import re
matches = [m.start() for m in re.finditer(r"template <> struct WGSMMA<3>", orig)]
print(f"WGSMMA<3> found {len(matches)} times")
matches_wg = [m.start() for m in re.finditer(r"template <> struct WG<3>", orig)]
print(f"WG<3> found {len(matches_wg)} times")

# Print the context around WGSMMA<3> in orig
for pos in matches:
    print("--- Context around WGSMMA<3> ---")
    print(orig[pos-100:pos+250])

for pos in matches_wg:
    print("--- Context around WG<3> ---")
    print(orig[pos-100:pos+250])
