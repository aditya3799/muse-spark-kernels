import torch

import submission

LOCAL_CONSOLIDATION_ENABLED = False


def route_pad(c: int) -> int:
    r, r2 = c & 255, c & 127
    return (c >> 8) * 256 + (r >> 7) * 128 + (32 if 0 < r2 <= 32 else 128 if r2 > 32 else 0)


def run_case(t: int, ls: int, seed: int) -> None:
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    # Small score spreads create many legal near ties while random routing leaves useful
    # mixed-W4 boundary/slack pairs among the 32 local experts.
    lg = (torch.randn(t, 512, device=dev) * 1.25).float()
    bias = (torch.randn(256, device=dev) * 0.004).float()
    local_start = torch.tensor([ls], dtype=torch.int32, device=dev)
    ids = torch.empty((t, 8), dtype=torch.int32, device=dev)
    loc = torch.empty_like(ids)
    lid = torch.empty_like(ids)
    tw = torch.empty((t, 8), dtype=torch.float32, device=dev)
    cnt = torch.empty(32, dtype=torch.int32, device=dev)
    cand = torch.empty((2, t), dtype=torch.int32, device=dev)
    candsc = torch.empty((2, t), dtype=torch.float32, device=dev)
    quota = torch.empty(3136, dtype=torch.int32, device=dev)

    score = torch.sigmoid((lg[:, :256] + lg[:, 256:]) * 0.1767766952966369) + bias
    ref_key, ref_ids = score.topk(9, dim=1, largest=True, sorted=True)
    raw = torch.sigmoid((lg[:, :256] + lg[:, 256:]) * 0.1767766952966369)
    base_ids = torch.empty_like(ids)
    base_loc_out, base_lid = torch.empty_like(ids), torch.empty_like(ids)
    base_tw = torch.empty_like(tw)
    submission._pf2().routing2(
        lg, bias, local_start, base_ids, base_loc_out, base_lid, base_tw,
        torch.empty(0, dtype=torch.int32, device=dev), cand, candsc, quota,
    )
    torch.cuda.synchronize()
    base_loc = (base_ids - ls) & 255
    base_cnt = torch.bincount(base_loc[base_loc < 32], minlength=32).to(torch.int32)

    submission._pf2().routing2(lg, bias, local_start, ids, loc, lid, tw, cnt, cand, candsc, quota)
    torch.cuda.synchronize()

    changed = ids != base_ids
    assert int(changed.sum(1).max()) <= 1
    rows, slots = changed.nonzero(as_tuple=True)
    if rows.numel():
        old = base_ids[rows, slots]
        new = ids[rows, slots]
        assert float((score[rows, old] - ref_key[rows, 8]).max()) <= 0.010001
        assert float((ref_key[rows, 7] - score[rows, new]).max()) <= 0.030001
    assert bool((ids.sort(1).values[:, 1:] != ids.sort(1).values[:, :-1]).all())
    expected_loc = (ids - ls) & 255
    assert torch.equal(loc, expected_loc)
    assert torch.equal(lid, torch.where(expected_loc < 32, expected_loc, -1))
    expected_tw = raw.gather(1, ids.long())
    expected_tw /= expected_tw.sum(1, keepdim=True)
    assert float((tw - expected_tw).abs().max()) < 2e-6
    assert float((tw.sum(1) - 1).abs().max()) < 2e-6
    assert torch.equal(cnt.cpu(), torch.bincount(lid[lid >= 0], minlength=32).to(torch.int32).cpu())

    plan = quota[1088:3136].cpu()
    amounts = plan[:1024].view(32, 32)
    local_moves = 0
    before_pad = sum(route_pad(int(v)) for v in base_cnt.cpu())
    planned_counts = base_cnt.cpu().clone()
    if LOCAL_CONSOLIDATION_ENABLED and t >= 16384:
        sources = set(torch.nonzero(amounts.sum(1), as_tuple=False).flatten().tolist())
        targets = set(torch.nonzero(amounts.sum(0), as_tuple=False).flatten().tolist())
        assert sources.isdisjoint(targets)
        for source in range(32):
            removed = int(amounts[source].sum())
            if not removed:
                continue
            assert int(planned_counts[source]) - removed > 0
            for target in range(32):
                amount = int(amounts[source, target])
                if not amount:
                    continue
                assert source != target
                assert route_pad(int(planned_counts[target]) + amount) == route_pad(int(planned_counts[target]))
                base_has_source = (base_loc == source).any(1)
                final_has_source = (expected_loc == source).any(1)
                base_has_target = (base_loc == target).any(1)
                final_has_target = (expected_loc == target).any(1)
                moved = (base_has_source & ~final_has_source & ~base_has_target & final_has_target).sum().item()
                assert moved == amount
                planned_counts[target] += amount
            planned_counts[source] -= removed
            local_moves += removed
    after_pad = sum(route_pad(int(v)) for v in planned_counts)
    assert after_pad <= before_pad
    assert set(torch.nonzero(base_cnt, as_tuple=False).flatten().tolist()) == set(torch.nonzero(planned_counts, as_tuple=False).flatten().tolist())
    assert int(planned_counts.sum()) == int(base_cnt.sum())

    print(
        f"routing probe T={t} ls={ls} OK:",
        f"changed={int(changed.sum())}",
        f"local_moves={local_moves}",
        f"planned_pad={before_pad}->{after_pad}",
        f"local_routes={int(base_cnt.sum())}->{int(cnt.sum())}",
    )


def main() -> None:
    cases = ((300, 0), (753, 32), (3012, 224), (4096, 0),
             (16384, 224), (16384, 0), (16384, 32), (16384, 96), (16384, 192))
    for i, (t, ls) in enumerate(cases):
        run_case(t, ls, 20261029 + i)


if __name__ == "__main__":
    main()
