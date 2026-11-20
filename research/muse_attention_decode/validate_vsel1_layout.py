"""Static lane/key checks for the q1 selective-V warp mapping."""

for wsub in range(2):
    covered = set()
    for lane in range(32):
        g, tig = lane >> 2, lane & 3
        for nt in range(4):
            for e in range(2):
                key = 32 * wsub + 8 * nt + 2 * tig + e
                covered.add((key, g))
    assert covered == {(key, g) for key in range(32 * wsub, 32 * wsub + 32) for g in range(8)}

    copies = set()
    for lane in range(32):
        for u in range(4):
            ve = lane + 32 * u
            key, chunk = 32 * wsub + (ve >> 2), ve & 3
            copies.add((key, chunk))
    assert copies == {(key, chunk) for key in range(32 * wsub, 32 * wsub + 32) for chunk in range(4)}

# Ballot low-nibble packing used in CUDA: owner lane=tig at g=0 maps to
# local key 8*nt + 2*tig + e exactly once.
for nt in range(4):
    for e in range(2):
        for nibble in range(16):
            mask = 0
            for tig in range(4):
                mask |= ((nibble >> tig) & 1) << (8 * nt + 2 * tig + e)
            for tig in range(4):
                assert ((mask >> (8 * nt + 2 * tig + e)) & 1) == ((nibble >> tig) & 1)

print("q1 selective-V lane, score, and copy coverage OK")
