"""Barrier-level deadlock check for the two-buffer XQA pipeline.

This models the order in which the IO, QK, and PV warpgroups reach each named
barrier.  It does not model instruction timing; it proves that every barrier
cycle has matching participants and that the finite schedule can drain.
"""

from __future__ import annotations


PARTICIPANTS = {
    "ioqk": frozenset(("io", "qk")),
    "iopv": frozenset(("io", "pv")),
    "qkpv": frozenset(("qk", "pv")),
    "reuse": frozenset(("io", "pv")),
}


def schedules(num_tiles: int) -> dict[str, list[tuple[str, int]]]:
    io: list[tuple[str, int]] = []
    qk: list[tuple[str, int]] = []
    pv: list[tuple[str, int]] = []
    for tile in range(num_tiles):
        buffer = tile & 1
        if tile >= 2:
            io.append(("reuse", buffer))
        io.append(("ioqk", buffer))
        io.append(("iopv", buffer))
        qk.append(("ioqk", buffer))
        qk.append(("qkpv", buffer))
    if num_tiles:
        pv.append(("iopv", 0))
    for tile in range(num_tiles):
        buffer = tile & 1
        pv.append(("qkpv", buffer))
        if tile + 1 < num_tiles:
            pv.append(("iopv", (tile + 1) & 1))
        if tile + 2 < num_tiles:
            pv.append(("reuse", buffer))
    return {"io": io, "qk": qk, "pv": pv}


def validate(num_tiles: int) -> int:
    work = schedules(num_tiles)
    cursor = {role: 0 for role in work}
    cycles = 0
    while any(cursor[role] < len(work[role]) for role in work):
        progressed = False
        for role, sequence in work.items():
            if cursor[role] >= len(sequence):
                continue
            event = sequence[cursor[role]]
            required = PARTICIPANTS[event[0]]
            if all(
                cursor[participant] < len(work[participant])
                and work[participant][cursor[participant]] == event
                for participant in required
            ):
                for participant in required:
                    cursor[participant] += 1
                cycles += 1
                progressed = True
                break
        if not progressed:
            waiting = {
                role: sequence[cursor[role]] if cursor[role] < len(sequence) else None
                for role, sequence in work.items()
            }
            raise AssertionError(f"deadlock for {num_tiles=} after {cycles} cycles: {waiting}")
    expected = 3 * num_tiles + max(0, num_tiles - 2)
    assert cycles == expected, (num_tiles, cycles, expected)
    return cycles


def main() -> None:
    for num_tiles in range(0, 4097):
        validate(num_tiles)
    print("pipeline barrier schedule OK for 0..4096 tiles")


if __name__ == "__main__":
    main()
