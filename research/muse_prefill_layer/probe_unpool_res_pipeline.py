import torch

import submission


torch.manual_seed(20261030)
dev = torch.device("cuda")
T, K, N = 4096, 4096, 8192

y = torch.randn((T, K), device=dev, dtype=torch.bfloat16)
w = torch.randn((N, K), device=dev, dtype=torch.bfloat16)
wt = w.t().contiguous()
f = torch.empty((T, N), device=dev, dtype=torch.bfloat16)
a = torch.randn((T, N), device=dev, dtype=torch.bfloat16)
s = torch.randn((T, N), device=dev, dtype=torch.float32)
ra = (0.5 + torch.rand((T,), device=dev)).float()
al1, be1, al2, be2, gf, gn = [torch.randn((N,), device=dev, dtype=torch.float32) for _ in range(6)]
so_ref, so_pipe = torch.empty_like(s), torch.empty_like(s)
h_ref, h_pipe = torch.empty_like(a), torch.empty_like(a)
flush = torch.empty((64 << 20,), device=dev, dtype=torch.float32)
g = submission._pf2()
cur, side = torch.cuda.current_stream(), torch.cuda.Stream()


def residual(lo, hi, so, hid):
    g.res_b(
        f[lo:hi], gf, a[lo:hi], ra[lo:hi], s[lo:hi], al1, be1, al2, be2, gn,
        so[lo:hi], hid[lo:hi],
    )


def baseline():
    torch.mm(y, wt, out=f)
    residual(0, T, so_ref, h_ref)


def pipeline(nchunk):
    step = (T + nchunk - 1) // nchunk
    for i in range(nchunk):
        lo, hi = i * step, min(T, (i + 1) * step)
        if lo >= hi:
            break
        torch.mm(y[lo:hi], wt, out=f[lo:hi])
        ready = torch.cuda.Event()
        ready.record(cur)
        with torch.cuda.stream(side):
            side.wait_event(ready)
            residual(lo, hi, so_pipe, h_pipe)
    cur.wait_stream(side)


baseline()
pipeline(2)
torch.cuda.synchronize()
sd = (so_ref - so_pipe).abs()
hd = (h_ref.float() - h_pipe.float()).abs()
print(
    "parity",
    "stream_max", float(sd.max()),
    "stream_rel", float(sd.norm() / so_ref.norm()),
    "hidden_max", float(hd.max()),
    "hidden_rel", float(hd.norm() / h_ref.float().norm()),
    "hidden_equal", bool(torch.equal(h_ref, h_pipe)),
)
assert torch.equal(so_ref, so_pipe)
assert torch.equal(h_ref, h_pipe)


def timing(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(11):
        flush.add_(1)
        beg, end = torch.cuda.Event(True), torch.cuda.Event(True)
        beg.record(cur)
        fn()
        end.record(cur)
        torch.cuda.synchronize()
        vals.append(beg.elapsed_time(end) * 1e3)
    vals.sort()
    return vals[len(vals) // 2]


tb = timing(baseline)
tmm = timing(lambda: torch.mm(y, wt, out=f))
tmm2 = timing(lambda: (torch.mm(y[: T // 2], wt, out=f[: T // 2]), torch.mm(y[T // 2 :], wt, out=f[T // 2 :])))
tr = timing(lambda: residual(0, T, so_ref, h_ref))
tr2 = timing(lambda: (residual(0, T // 2, so_ref, h_ref), residual(T // 2, T, so_ref, h_ref)))
t2 = timing(lambda: pipeline(2))
t4 = timing(lambda: pipeline(4))
print(
    f"local T{T} us: gemm={tmm:.1f} gemm2={tmm2:.1f} resb={tr:.1f} resb2={tr2:.1f} "
    f"dense={tb:.1f} pipe2={t2:.1f} pipe4={t4:.1f} ratios={t2/tb:.3f}/{t4/tb:.3f}"
)
