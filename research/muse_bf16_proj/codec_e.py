"""Lossless entropy coding for the byte-sized C12 high-code plane.

Codec E does not change C12's numerical representation.  It only entropy-codes
pairs of four-bit C12 codes (one byte/symbol), so the existing C12 decoder and
escape correction remain byte-identical after this stream is unpacked.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import torch


_NSYMBOLS = 256
_NPREFIX = 8
_WORD_BITS = 32
_WORD_MASK = (1 << _WORD_BITS) - 1


@dataclass(frozen=True)
class CodecEPlan:
    """A canonical three-bit-prefix code for the 256 byte symbols."""

    suffix_bits: tuple[int, ...]
    bases: tuple[int, ...]
    lut: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.suffix_bits) != _NPREFIX or len(self.bases) != _NPREFIX:
            raise ValueError("Codec E requires eight prefix groups")
        if len(self.lut) != _NSYMBOLS or sorted(self.lut) != list(range(_NSYMBOLS)):
            raise ValueError("Codec E LUT must be a permutation of all byte symbols")
        if any(bits < 0 or bits > 8 for bits in self.suffix_bits):
            raise ValueError("suffix widths must be in [0, 8]")
        for group, (base, bits) in enumerate(zip(self.bases, self.suffix_bits)):
            end = self.bases[group + 1] if group + 1 < _NPREFIX else _NSYMBOLS
            if base < 0 or base >= _NSYMBOLS or end <= base:
                raise ValueError("every prefix group must contain at least one symbol")
            if end - base > 1 << bits:
                raise ValueError("prefix group exceeds its suffix capacity")

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(3 + bits for bits in self.suffix_bits)

    def tensors(self, device: torch.device | str) -> tuple[torch.Tensor, ...]:
        lut = torch.tensor(self.lut, dtype=torch.long, device=device)
        inverse = torch.empty_like(lut)
        inverse[lut] = torch.arange(_NSYMBOLS, dtype=torch.long, device=device)
        suffix_bits = torch.tensor(self.suffix_bits, dtype=torch.long, device=device)
        bases = torch.tensor(self.bases, dtype=torch.long, device=device)
        rank_group = torch.empty(_NSYMBOLS, dtype=torch.long, device=device)
        for group, base in enumerate(self.bases):
            end = self.bases[group + 1] if group + 1 < _NPREFIX else _NSYMBOLS
            rank_group[base:end] = group
        return lut, inverse, suffix_bits, bases, rank_group


@dataclass
class PackedCodecE:
    """Word-aligned independent Codec E streams.

    ``offsets`` and ``bit_counts`` have shape ``stream_shape + (substreams,)``.
    The flattened stream order is original-stream major, then substream.
    """

    words: torch.Tensor
    offsets: torch.Tensor
    bit_counts: torch.Tensor
    symbols_per_stream: int
    substreams: int
    stream_shape: tuple[int, ...]
    plan: CodecEPlan

    @property
    def payload_bits(self) -> int:
        return int(self.bit_counts.sum().item())

    @property
    def allocated_bits(self) -> int:
        # One final sentinel word is excluded.
        return max(0, self.words.numel() - 1) * _WORD_BITS

    @property
    def bits_per_symbol(self) -> float:
        n = 1
        for dim in self.stream_shape:
            n *= dim
        return self.payload_bits / (n * self.symbols_per_stream)


@dataclass
class WarpInterleavedCodecE:
    """Codec E words transposed as ``[warp][substream][slot][lane]``.

    The final two source dimensions are ``[32 lanes, symbols]``.  Each segment
    is padded only to the longest lane in that warp/substream, making every
    refill a contiguous 128-byte warp transaction while keeping padding small.
    """

    words: torch.Tensor
    offsets: torch.Tensor
    bit_counts: torch.Tensor
    word_counts: torch.Tensor
    symbols_per_stream: int
    substreams: int
    warp_shape: tuple[int, ...]
    plan: CodecEPlan

    @property
    def payload_bits(self) -> int:
        return int(self.bit_counts.sum().item())

    @property
    def allocated_bits(self) -> int:
        # A final 32-word warp sentinel is excluded.
        return max(0, self.words.numel() - 32) * _WORD_BITS

    @property
    def bits_per_symbol(self) -> float:
        warps = 1
        for dim in self.warp_shape:
            warps *= dim
        return self.payload_bits / (warps * 32 * self.symbols_per_stream)


def symbol_histogram(symbols: torch.Tensor) -> torch.Tensor:
    """Return the exact 256-bin histogram of a byte tensor on the CPU."""

    if symbols.dtype != torch.uint8:
        raise TypeError("Codec E symbols must have dtype torch.uint8")
    return torch.bincount(symbols.reshape(-1).long(), minlength=_NSYMBOLS).cpu()


def _optimal_suffix_bits(sorted_counts: tuple[int, ...]) -> tuple[int, ...]:
    """Minimize payload bits for eight ordered power-of-two-capacity groups."""

    prefix = [0]
    for count in sorted_counts:
        prefix.append(prefix[-1] + count)

    @lru_cache(maxsize=None)
    def solve(group: int, rank: int):
        groups_left = _NPREFIX - group
        symbols_left = _NSYMBOLS - rank
        if groups_left == 0:
            return (0, ()) if symbols_left == 0 else None
        if symbols_left < groups_left:
            return None
        best = None
        for bits in range(9):
            capacity = 1 << bits
            take = min(capacity, symbols_left - (groups_left - 1))
            if take <= 0:
                continue
            child = solve(group + 1, rank + take)
            if child is None:
                continue
            weighted = (prefix[rank + take] - prefix[rank]) * (3 + bits)
            candidate = (weighted + child[0], (bits,) + child[1])
            # Deterministic ties prefer smaller early widths and then smaller
            # total capacity, which reduces unused suffix values.
            if best is None or candidate < best:
                best = candidate
        return best

    result = solve(0, 0)
    if result is None:
        raise RuntimeError("failed to construct Codec E prefix groups")
    return result[1]


def build_plan(symbols_or_histogram: torch.Tensor) -> CodecEPlan:
    """Build a deterministic frequency-ranked Codec E plan.

    The argument may be a byte tensor of symbols or a 256-entry histogram.
    Symbol-value order breaks frequency ties, so independently repeated packs
    produce identical LUTs.
    """

    if symbols_or_histogram.dtype == torch.uint8:
        hist = symbol_histogram(symbols_or_histogram)
    else:
        hist = symbols_or_histogram.reshape(-1).to(dtype=torch.long, device="cpu")
        if hist.numel() != _NSYMBOLS or bool((hist < 0).any()):
            raise ValueError("histogram must contain 256 non-negative counts")
    counts = [int(v) for v in hist.tolist()]
    lut = tuple(sorted(range(_NSYMBOLS), key=lambda symbol: (-counts[symbol], symbol)))
    sorted_counts = tuple(counts[symbol] for symbol in lut)
    suffix_bits = _optimal_suffix_bits(sorted_counts)
    bases = []
    rank = 0
    for group, bits in enumerate(suffix_bits):
        bases.append(rank)
        groups_left = _NPREFIX - group
        symbols_left = _NSYMBOLS - rank
        rank += min(1 << bits, symbols_left - (groups_left - 1))
    if rank != _NSYMBOLS:
        raise AssertionError("Codec E group construction did not cover the LUT")
    return CodecEPlan(tuple(suffix_bits), tuple(bases), lut)


def pack(symbols: torch.Tensor, plan: CodecEPlan | None = None, *, substreams: int = 1) -> PackedCodecE:
    """Pack byte symbols whose final dimension is one independent lane stream.

    Symbols are round-robin distributed across 1, 2, or 4 independent streams.
    Every substream begins on a 32-bit boundary, matching the intended refill
    ring.  The returned word tensor includes one zero sentinel word so a decoder
    may safely form a two-word refill window at the end of the final stream.
    """

    if symbols.dtype != torch.uint8:
        raise TypeError("Codec E symbols must have dtype torch.uint8")
    if substreams not in (1, 2, 4):
        raise ValueError("Codec E supports 1, 2, or 4 substreams")
    if symbols.ndim < 1 or symbols.shape[-1] == 0 or symbols.shape[-1] % substreams:
        raise ValueError("the final dimension must be non-empty and divisible by substreams")
    if plan is None:
        plan = build_plan(symbols)

    stream_shape = tuple(symbols.shape[:-1])
    symbols_per_stream = symbols.shape[-1]
    per_substream = symbols_per_stream // substreams
    split = torch.stack([symbols[..., lane::substreams] for lane in range(substreams)], dim=-2)
    flat = split.reshape(-1, per_substream).long()
    lut, inverse, suffix_bits, bases, rank_group = plan.tensors(symbols.device)
    del lut
    ranks = inverse[flat]
    groups = rank_group[ranks]
    lengths = 3 + suffix_bits[groups]
    suffixes = ranks - bases[groups]
    codewords = groups | (suffixes << 3)
    bit_positions = torch.cumsum(lengths, dim=1) - lengths
    bit_counts = lengths.sum(dim=1)
    word_counts = (bit_counts + (_WORD_BITS - 1)) // _WORD_BITS
    offsets = torch.cumsum(word_counts, dim=0) - word_counts
    total_words = int(word_counts.sum().item())

    global_bits = offsets[:, None] * _WORD_BITS + bit_positions
    word_index = global_bits >> 5
    shifts = global_bits & 31
    accumulator = torch.zeros(total_words + 1, dtype=torch.long, device=symbols.device)
    low = (codewords << shifts) & _WORD_MASK
    accumulator.index_add_(0, word_index.reshape(-1), low.reshape(-1))
    crosses = shifts + lengths > _WORD_BITS
    if bool(crosses.any()):
        high = torch.where(crosses, codewords >> (_WORD_BITS - shifts), 0)
        accumulator.index_add_(0, (word_index + 1).reshape(-1), high.reshape(-1))
    if bool((accumulator > _WORD_MASK).any()):
        raise AssertionError("overlapping Codec E fields corrupted a packed word")
    words = accumulator.to(torch.int32)
    out_shape = stream_shape + (substreams,)
    return PackedCodecE(
        words=words,
        offsets=offsets.to(torch.int32).reshape(out_shape).contiguous(),
        bit_counts=bit_counts.to(torch.int32).reshape(out_shape).contiguous(),
        symbols_per_stream=symbols_per_stream,
        substreams=substreams,
        stream_shape=stream_shape,
        plan=plan,
    )


def unpack(packed: PackedCodecE) -> torch.Tensor:
    """Independent reference decoder for :func:`pack`."""

    plan = packed.plan
    device = packed.words.device
    lut, _inverse, suffix_bits, bases, _rank_group = plan.tensors(device)
    words = packed.words.long() & _WORD_MASK
    offsets = packed.offsets.reshape(-1).long()
    bit_counts = packed.bit_counts.reshape(-1).long()
    per_substream = packed.symbols_per_stream // packed.substreams
    bit_positions = torch.zeros_like(offsets)
    decoded = torch.empty((offsets.numel(), per_substream), dtype=torch.uint8, device=device)

    for index in range(per_substream):
        absolute = offsets * _WORD_BITS + bit_positions
        word_index = absolute >> 5
        shift = absolute & 31
        low_part = words[word_index] >> shift
        # Mask before shifting so the reference stays inside signed int64 even
        # when shift == 0 (the masked value is then zero).
        high_mask = (1 << shift) - 1
        high_part = (words[word_index + 1] & high_mask) << (_WORD_BITS - shift)
        window = low_part | high_part
        groups = window & 7
        widths = suffix_bits[groups]
        masks = (1 << widths) - 1
        ranks = bases[groups] + ((window >> 3) & masks)
        if bool((ranks >= _NSYMBOLS).any()):
            raise ValueError("invalid Codec E suffix/LUT index")
        decoded[:, index] = lut[ranks].to(torch.uint8)
        bit_positions += 3 + widths

    if not torch.equal(bit_positions, bit_counts):
        raise ValueError("Codec E stream length does not match its symbol count")
    split = decoded.reshape(packed.stream_shape + (packed.substreams, per_substream))
    result = torch.empty(packed.stream_shape + (packed.symbols_per_stream,), dtype=torch.uint8, device=device)
    for lane in range(packed.substreams):
        result[..., lane::packed.substreams] = split[..., lane, :]
    return result


def pack_warp_interleaved(symbols: torch.Tensor, plan: CodecEPlan | None = None, *,
                          substreams: int = 1) -> WarpInterleavedCodecE:
    """Pack lane streams into coalesced ``[substream][slot][lane]`` words."""

    if symbols.dtype != torch.uint8:
        raise TypeError("Codec E symbols must have dtype torch.uint8")
    if symbols.ndim < 2 or symbols.shape[-2] != 32:
        raise ValueError("the penultimate dimension must contain exactly 32 warp lanes")
    if substreams not in (1, 2, 4):
        raise ValueError("Codec E supports 1, 2, or 4 substreams")
    if symbols.shape[-1] == 0 or symbols.shape[-1] % substreams:
        raise ValueError("the symbol count must be non-empty and divisible by substreams")
    if plan is None:
        plan = build_plan(symbols)

    warp_shape = tuple(symbols.shape[:-2])
    symbols_per_stream = symbols.shape[-1]
    per_substream = symbols_per_stream // substreams
    warps = symbols.reshape(-1, 32, symbols_per_stream)
    split = torch.stack([warps[..., lane::substreams] for lane in range(substreams)], dim=1)
    # [warp, substream, lane, sequential symbols]
    lut, inverse, suffix_bits, bases, rank_group = plan.tensors(symbols.device)
    del lut
    ranks = inverse[split.long()]
    groups = rank_group[ranks]
    lengths = 3 + suffix_bits[groups]
    suffixes = ranks - bases[groups]
    codewords = groups | (suffixes << 3)
    bit_positions = torch.cumsum(lengths, dim=-1) - lengths
    bit_counts = lengths.sum(dim=-1)
    word_counts = (bit_counts + (_WORD_BITS - 1)) // _WORD_BITS
    max_words = word_counts.amax(dim=-1)
    segment_words = max_words * 32
    offsets = torch.cumsum(segment_words.reshape(-1), dim=0) - segment_words.reshape(-1)
    offsets = offsets.reshape(warps.shape[0], substreams)
    total_words = int(segment_words.sum().item())

    lane_index = torch.arange(32, dtype=torch.long, device=symbols.device).view(1, 1, 32, 1)
    word_index = bit_positions >> 5
    shifts = bit_positions & 31
    destinations = offsets[:, :, None, None] + word_index * 32 + lane_index
    accumulator = torch.zeros(total_words + 32, dtype=torch.long, device=symbols.device)
    low = (codewords << shifts) & _WORD_MASK
    accumulator.index_add_(0, destinations.reshape(-1), low.reshape(-1))
    crosses = shifts + lengths > _WORD_BITS
    if bool(crosses.any()):
        high = torch.where(crosses, codewords >> (_WORD_BITS - shifts), 0)
        accumulator.index_add_(0, (destinations + 32).reshape(-1), high.reshape(-1))
    if bool((accumulator > _WORD_MASK).any()):
        raise AssertionError("overlapping warp-interleaved Codec E fields corrupted a word")
    return WarpInterleavedCodecE(
        words=accumulator.to(torch.int32),
        offsets=offsets.to(torch.int32).reshape(warp_shape + (substreams,)).contiguous(),
        bit_counts=bit_counts.to(torch.int32).reshape(warp_shape + (substreams, 32)).contiguous(),
        word_counts=word_counts.to(torch.int32).reshape(warp_shape + (substreams, 32)).contiguous(),
        symbols_per_stream=symbols_per_stream,
        substreams=substreams,
        warp_shape=warp_shape,
        plan=plan,
    )


def unpack_warp_interleaved(packed: WarpInterleavedCodecE) -> torch.Tensor:
    """Independent decoder for the warp-interleaved storage layout."""

    device = packed.words.device
    lut, _inverse, suffix_bits, bases, _rank_group = packed.plan.tensors(device)
    words = packed.words.long() & _WORD_MASK
    offsets = packed.offsets.reshape(-1, packed.substreams).long()
    bit_counts = packed.bit_counts.reshape(-1, packed.substreams, 32).long()
    warps = offsets.shape[0]
    per_substream = packed.symbols_per_stream // packed.substreams
    positions = torch.zeros((warps, packed.substreams, 32), dtype=torch.long, device=device)
    decoded = torch.empty((warps, packed.substreams, 32, per_substream), dtype=torch.uint8, device=device)
    lanes = torch.arange(32, dtype=torch.long, device=device).view(1, 1, 32)

    for index in range(per_substream):
        word_index = positions >> 5
        shift = positions & 31
        addresses = offsets[:, :, None] + word_index * 32 + lanes
        low_part = words[addresses] >> shift
        high_mask = (1 << shift) - 1
        high_part = (words[addresses + 32] & high_mask) << (_WORD_BITS - shift)
        window = low_part | high_part
        groups = window & 7
        widths = suffix_bits[groups]
        ranks = bases[groups] + ((window >> 3) & ((1 << widths) - 1))
        if bool((ranks >= _NSYMBOLS).any()):
            raise ValueError("invalid warp-interleaved Codec E suffix/LUT index")
        decoded[..., index] = lut[ranks].to(torch.uint8)
        positions += 3 + widths

    if not torch.equal(positions, bit_counts):
        raise ValueError("warp-interleaved stream length does not match its symbol count")
    result = torch.empty((warps, 32, packed.symbols_per_stream), dtype=torch.uint8, device=device)
    for lane in range(packed.substreams):
        result[..., lane::packed.substreams] = decoded[:, lane]
    return result.reshape(packed.warp_shape + (32, packed.symbols_per_stream))


def expected_bits(histogram: torch.Tensor, plan: CodecEPlan) -> int:
    """Return the exact payload-bit count implied by a symbol histogram."""

    hist = histogram.reshape(-1).to(dtype=torch.long, device="cpu")
    if hist.numel() != _NSYMBOLS:
        raise ValueError("histogram must contain 256 entries")
    total = 0
    for group, base in enumerate(plan.bases):
        end = plan.bases[group + 1] if group + 1 < _NPREFIX else _NSYMBOLS
        for rank in range(base, end):
            total += int(hist[plan.lut[rank]]) * (3 + plan.suffix_bits[group])
    return total


def c12_pair_symbols(weights: torch.Tensor):
    """Create the ordinary row-major C12 planes used for codec validation.

    Returns ``(low_bytes, pair_symbols, base, escape_indices, original_bits)``.
    Applying the escapes after unpacking the pair symbols reconstructs every
    original BF16 coefficient bit-for-bit.
    """

    if weights.dtype != torch.bfloat16 or weights.shape[-1] % 2:
        raise ValueError("weights must be BF16 with an even final dimension")
    bits = weights.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    high = bits >> 8
    exponent = high & 0x7F
    hist = torch.bincount(exponent.reshape(-1).long(), minlength=128)
    windows = torch.stack([hist[base:base + 8].sum() for base in range(121)])
    base = int(windows.argmax())
    offset = (exponent - base).clamp(0, 7)
    codes = (((high >> 7) << 3) | offset).to(torch.uint8)
    pair_symbols = codes[..., 0::2] | (codes[..., 1::2] << 4)
    escapes = ((exponent < base) | (exponent > base + 7)).nonzero()
    return (bits & 0xFF).to(torch.uint8), pair_symbols.contiguous(), base, escapes, bits


def reconstruct_c12(low: torch.Tensor, pair_symbols: torch.Tensor, base: int,
                    escapes: torch.Tensor, original_bits: torch.Tensor) -> torch.Tensor:
    """Reconstruct BF16 bits from C12 planes and apply exact escapes."""

    codes = torch.empty_like(low)
    codes[..., 0::2] = pair_symbols & 0xF
    codes[..., 1::2] = pair_symbols >> 4
    high = ((codes >> 3).to(torch.int32) << 7) | (base + (codes & 7).to(torch.int32))
    decoded = (high << 8) | low.to(torch.int32)
    if escapes.numel():
        index = tuple(escapes.t())
        decoded[index] = original_bits[index]
    return decoded


def format_plan(plan: CodecEPlan) -> str:
    capacities: Iterable[int] = (1 << bits for bits in plan.suffix_bits)
    return f"suffix_bits={plan.suffix_bits}, bases={plan.bases}, capacities={tuple(capacities)}"
