import torch


torch.manual_seed(19)
dev = "cuda"
T, K, NP, NR = 3012, 8192, 4096, 512
h = torch.randn((T, K), device=dev, dtype=torch.bfloat16)
wp = torch.randn((NP, K), device=dev, dtype=torch.bfloat16)
wr = torch.randn((NR, K), device=dev, dtype=torch.bfloat16)
wf = torch.cat((wp, wr), 0).contiguous()
flush = torch.empty((64 << 20) // 4, device=dev, dtype=torch.float32)
sink = torch.empty((), device=dev)
side = torch.cuda.Stream()


def current():
    cur = torch.cuda.current_stream()
    side.wait_stream(cur)
    with torch.cuda.stream(side):
        lg = torch.mm(h, wr.t(), out_dtype=torch.float32)
    pr = h @ wp.t()
    cur.wait_stream(side)
    return pr, lg


def fused():
    z = torch.mm(h, wf.t(), out_dtype=torch.float32)
    return z[:, :NP].to(torch.bfloat16), z[:, NP:]


pr0, lg0 = current()
pr1, lg1 = fused()
torch.cuda.synchronize()
print(
    "parity",
    "pool_equal", bool(torch.equal(pr0, pr1)),
    "pool_rel", float((pr0.float() - pr1.float()).norm() / pr0.float().norm()),
    "router_rel", float((lg0 - lg1).norm() / lg0.norm()),
    "router_max", float((lg0 - lg1).abs().max()),
)


def capture(fn):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    return g


def timing(g):
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    vals = []
    for _ in range(11):
        sink.copy_(flush.sum())
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        vals.append(a.elapsed_time(b) * 1000.0)
    vals.sort()
    return vals[len(vals) // 2], (vals[8] - vals[2]) / 2


g0 = capture(current)
g1 = capture(fused)
t0, s0 = timing(g0)
t1, s1 = timing(g1)
print(f"cold graph us: current={t0:.1f}+-{s0:.1f} fused={t1:.1f}+-{s1:.1f} ratio={t1/t0:.3f}")
