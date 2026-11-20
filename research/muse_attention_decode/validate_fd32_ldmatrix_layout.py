"""Symbolically verify the FD32 ldmatrix accumulator/output permutation."""


def old_dimension(t: int, n: int, c_col: int) -> int:
    """Dimension represented by the old MMA n and live C column."""
    return 8 * (2 * t + c_col) + n


def ldmt_dimension(t: int, mma: int, c_col: int) -> int:
    """Dimension represented by ldmatrix MMA {matrix, dimension parity}."""
    matrix, parity = divmod(mma, 2)
    return 16 * matrix + 4 * t + 2 * c_col + parity


old_slots = {}
for t in range(4):
    for c_col in range(2):
        for n in range(8):
            d = old_dimension(t, n, c_col)
            old_slots[d] = (4 * t + 2 * c_col + n // 4, n % 4)

ldmt_slots = {}
for t in range(4):
    for matrix in range(4):
        unit = 4 * matrix + t
        for parity in range(2):
            mma = 2 * matrix + parity
            for c_col in range(2):
                d = ldmt_dimension(t, mma, c_col)
                ldmt_slots[d] = (unit, 2 * c_col + parity)

assert set(old_slots) == set(range(64))
assert set(ldmt_slots) == set(range(64))
for dimension in range(64):
    old_unit, old_float4_lane = old_slots[dimension]
    new_unit, new_float4_lane = ldmt_slots[dimension]
    assert 4 * old_unit + old_float4_lane == dimension
    assert 4 * new_unit + new_float4_lane == dimension

print("FD32 ldmatrix layout: all 64 output dimensions map exactly")
