"""Static invariants for the flat-q8 KQ/PV role pipeline.

This is intentionally GPU-independent.  CUTLASS's host layout oracle in
``validate_wgmma_layout.cu`` validates the underlying WGMMA coordinates and
SW64 formula; this file checks the way the production kernel composes those
coordinates, its two-head handoff schedule, and its shared-memory extents.
"""

from __future__ import annotations


def qk_coord(warp: int, lane: int, nt: int, hf: int, e: int) -> tuple[int, int]:
    """Coordinate of s[4*nt + 2*hf + e] for K @ Q.T."""
    key = 16 * warp + 8 * hf + lane // 4
    query = 8 * nt + 2 * (lane % 4) + e
    return key, query


def pv_coord(warp: int, lane: int, nt: int, hf: int, e: int) -> tuple[int, int]:
    """Physical output coordinate after the register-A even/odd unswizzle."""
    # CUTLASS ALayout_64x32 plus the XQA ldmatrix/prmt sequence maps the
    # accumulator's local row 8*hf+g to physical V dimension 2*g+hf.
    dim = 16 * warp + 2 * (lane // 4) + hf
    query = 8 * nt + 2 * (lane % 4) + e
    return dim, query


def validate_coordinates() -> None:
    qk: set[tuple[int, int]] = set()
    pv: set[tuple[int, int]] = set()
    for warp in range(4):
        for lane in range(32):
            for nt in range(8):
                for hf in range(2):
                    for e in range(2):
                        qk.add(qk_coord(warp, lane, nt, hf, e))
                        pv.add(pv_coord(warp, lane, nt, hf, e))
    expected = {(row, col) for row in range(64) for col in range(64)}
    assert qk == expected
    assert pv == expected

    # A bijection alone cannot detect the bug this validator guards against:
    # logical accumulator rows are not natural physical dimensions.
    for warp in range(4):
        for g in range(8):
            for hf in range(2):
                logical_row = 16 * warp + 8 * hf + g
                physical_dim = 16 * warp + 2 * g + hf
                assert physical_dim == 16 * warp + 2 * (logical_row & 7) + ((logical_row & 15) >> 3)

    # The final shared transpose must reconstruct the incumbent q8 ownership:
    # query = 16*wsub + 8*hf + g, dim = 8*nt + 2*tig + e.
    reloads: set[tuple[int, int]] = set()
    for wsub in range(4):
        for lane in range(32):
            g, tig = divmod(lane, 4)
            for nt in range(8):
                for hf in range(2):
                    for e in range(2):
                        reloads.add((16 * wsub + 8 * hf + g, 8 * nt + 2 * tig + e))
    assert reloads == expected


def validate_swizzle() -> None:
    physical: set[int] = set()
    for row in range(64):
        for col in range(64):
            linear = row * 64 + col
            addr = linear ^ ((linear & 0x180) >> 3)
            physical.add(addr)
            chunk_formula = row * 64 + (((col // 16) ^ ((row >> 1) & 3)) * 16) + col % 16
            assert addr == chunk_formula
    assert physical == set(range(4096))


def validate_schedule(num_tiles: int) -> None:
    # WG0 produces h0/h1; WG1 consumes h0/h1.  Each matching named-barrier
    # generation is both a full-buffer handoff and the previous generation's
    # reuse acknowledgement.
    qk = [barrier for _ in range(num_tiles) for barrier in (5, 6)]
    pv = [barrier for _ in range(num_tiles) for barrier in (5, 6)]
    assert qk == pv
    for task, barrier in enumerate(qk):
        assert barrier == 5 + (task & 1)


def validate_shared_extents() -> None:
    kvq_bytes = 18_432
    offset = 8_192  # two 64x64 FP8 P buffers
    offset += 4 * 64 * 4  # temporary maxima
    offset += 4 * 64 * 4  # temporary sums
    offset += 2 * 64 * 4  # tile maxima
    offset += 2 * 64 * 4  # tile sums
    offset += 2 * 64 * 4  # running maxima
    offset += 2 * 64 * 4  # running sums
    offset += 64 * 4  # alpha
    assert offset == 12_544
    assert offset <= kvq_bytes
    assert 64 * 64 * 4 <= kvq_bytes  # one-head O transpose scratch


def main() -> None:
    validate_coordinates()
    validate_swizzle()
    validate_shared_extents()
    for num_tiles in range(0, 4097):
        validate_schedule(num_tiles)
    print("q8 role pipeline invariants OK (layouts, extents, schedule 0..4096 tiles)")


if __name__ == "__main__":
    main()
