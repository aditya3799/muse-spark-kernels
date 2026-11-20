import torch

from codec_e import (
    build_plan,
    c12_pair_symbols,
    expected_bits,
    pack,
    pack_warp_interleaved,
    reconstruct_c12,
    symbol_histogram,
    unpack,
    unpack_warp_interleaved,
)


def _devices():
    yield "cpu"
    if torch.cuda.is_available():
        yield "cuda"


def test_all_symbols_roundtrip():
    symbols = torch.arange(256, dtype=torch.uint8).repeat(7, 4)
    plan = build_plan(symbols)
    for device in _devices():
        source = symbols.to(device)
        for substreams in (1, 2, 4):
            encoded = pack(source, plan, substreams=substreams)
            assert torch.equal(unpack(encoded), source)


def test_skewed_random_roundtrip_and_accounting():
    generator = torch.Generator().manual_seed(147)
    # A deliberately skewed distribution resembles the C12 high-code plane.
    probabilities = torch.arange(256, 0, -1, dtype=torch.float64)
    probabilities /= probabilities.sum()
    symbols = torch.multinomial(probabilities, 31 * 128, replacement=True, generator=generator).to(torch.uint8).view(31, 128)
    plan = build_plan(symbols)
    histogram = symbol_histogram(symbols)
    for substreams in (1, 2, 4):
        encoded = pack(symbols, plan, substreams=substreams)
        assert torch.equal(unpack(encoded), symbols)
        assert encoded.payload_bits == expected_bits(histogram, plan)
        assert encoded.allocated_bits >= encoded.payload_bits
        assert encoded.allocated_bits - encoded.payload_bits < symbols.shape[0] * substreams * 32


def test_c12_reconstruction_is_bit_exact():
    generator = torch.Generator().manual_seed(5151)
    weights = (torch.randn((33, 1024), generator=generator) / (1024 ** 0.5)).to(torch.bfloat16)
    # Include exponent-window outliers and signed zeros to exercise escapes.
    raw = weights.view(torch.int16)
    raw[0, :8] = torch.tensor([0x0000, -32768, 0x0001, -32767, 0x7F7F, -129, 0x0080, -32640], dtype=torch.int16)
    low, symbols, base, escapes, original = c12_pair_symbols(weights)
    plan = build_plan(symbols)
    for substreams in (1, 2, 4):
        encoded = pack(symbols, plan, substreams=substreams)
        decoded_symbols = unpack(encoded)
        recovered = reconstruct_c12(low, decoded_symbols, base, escapes, original)
        assert torch.equal(recovered, original)


def test_plan_covers_every_rank_once():
    histogram = torch.arange(256, dtype=torch.long) ** 2 + 1
    plan = build_plan(histogram)
    assert sorted(plan.lut) == list(range(256))
    assert plan.bases[0] == 0
    for group, base in enumerate(plan.bases):
        end = plan.bases[group + 1] if group + 1 < 8 else 256
        assert 1 <= end - base <= 1 << plan.suffix_bits[group]


def test_warp_interleaved_roundtrip_and_padding():
    generator = torch.Generator().manual_seed(1818)
    symbols = torch.randint(0, 256, (3, 5, 32, 128), dtype=torch.uint8, generator=generator)
    # Make the distribution nonuniform so lane word counts really differ.
    symbols &= torch.randint(0, 256, symbols.shape, dtype=torch.uint8, generator=generator)
    plan = build_plan(symbols)
    for device in _devices():
        source = symbols.to(device)
        for substreams in (1, 2, 4):
            encoded = pack_warp_interleaved(source, plan, substreams=substreams)
            assert torch.equal(unpack_warp_interleaved(encoded), source)
            assert encoded.payload_bits == expected_bits(symbol_histogram(source), plan)
            assert encoded.allocated_bits >= encoded.payload_bits
