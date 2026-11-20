"""Search for a cheaper exact E2M1 packed-code decode layout.

Each decoded half2 register wants code bits in positions {6, 7, 8, 15},
duplicated in the upper 16-bit half.  A register can be decoded with one
rotate plus one mask iff its eight packed source positions are a translation
of that pattern in Z/32Z.  Four pairwise-disjoint translations would cover
all 32 input bits and eliminate the multiply/split fourth-register decode.
"""

from itertools import combinations


WIDTH = 32
TARGET = frozenset((6, 7, 8, 15, 22, 23, 24, 31))


def translated(offset: int) -> frozenset[int]:
    return frozenset((bit + offset) % WIDTH for bit in TARGET)


def main() -> None:
    sets = [translated(offset) for offset in range(WIDTH)]

    # Translation symmetry lets us fix one offset at zero.
    tilings: list[tuple[int, int, int, int]] = []
    for others in combinations(range(1, WIDTH), 3):
        offsets = (0, *others)
        chosen = [sets[offset] for offset in offsets]
        if len(set().union(*chosen)) == WIDTH:
            tilings.append(offsets)

    best_offsets: tuple[int, ...] = ()
    best_covered = 0
    for count in range(1, 5):
        for others in combinations(range(1, WIDTH), count - 1):
            offsets = (0, *others)
            covered = len(set().union(*(sets[offset] for offset in offsets)))
            if covered > best_covered:
                best_covered = covered
                best_offsets = offsets

    print(f"target={sorted(TARGET)}")
    print(f"four-translation tilings={len(tilings)}")
    if tilings:
        for offsets in tilings[:16]:
            print(f"tiling offsets={offsets}")
    print(
        "best normalized translate family: "
        f"offsets={best_offsets}, covered={best_covered}/{WIDTH}"
    )
    print(
        "sets="
        + "; ".join(
            f"{offset}:{sorted(sets[offset])}" for offset in best_offsets
        )
    )


if __name__ == "__main__":
    main()
