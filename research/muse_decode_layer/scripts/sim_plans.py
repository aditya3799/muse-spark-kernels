sms = 132 # H200 SM count
target = 2 * sms # 264

shapes = [
    ("decode 1 sliding", 1, 1, True),
    ("decode 1 full", 1, 1, False),
    ("decode 16 sliding", 16, 1, True),
    ("decode 16 full", 16, 1, False),
    ("decode 32 sliding", 32, 1, True),
    ("decode 32 full", 32, 1, False),
    ("verify 1 sliding", 1, 8, True),
    ("verify 1 full", 1, 8, False),
    ("verify 16 sliding", 16, 8, True),
    ("verify 16 full", 16, 8, False),
    ("verify 32 sliding", 32, 8, True),
    ("verify 32 full", 32, 8, False),
]

_TUNED7 = {
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
}

def sim_plan(name, B, L, window):
    pairs = 1
    P = 64
    max_pages = 64
    tuned = _TUNED7.get((B, L, bool(window), pairs))
    if tuned:
        print(f"{name:<20}: TUNED override: {tuned}")
        return
    
    if not window and 1 < B <= 512:
        ft = (3 * sms) if L == 1 else target
        print(f"{name:<20}: FLAT: n_splits={ft}, flat=1, nc=1")
        return
        
    keys_max = min(4096 + L - 1, max_pages * P) if window else max_pages * P
    tiles = max(1, (keys_max + 63) // 64)
    reqs = B * pairs
    
    if not window:
        if L == 8:
            n = max(1, min(tiles, ((target * 21) // 22) // reqs))
            print(f"{name:<20}: PR-FULL: n_splits={n}, flat=0, nc={max(1, min(n, 8))}")
            return
        else:
            n = max(1, min(tiles, target // reqs))
            print(f"{name:<20}: PR-FULL: n_splits={n}, flat=0, nc=1")
            return
            
    # window is True:
    if L == 8:
        # sliding L=8
        n = max(1, min(tiles, target // reqs))
        print(f"{name:<20}: PR-SLIDING L=8: n_splits={n}, tiles={tiles}, reqs={reqs}")
    else:
        # sliding L=1
        n = max(1, min(tiles, target // reqs))
        print(f"{name:<20}: PR-SLIDING L=1: n_splits={n}, tiles={tiles}, reqs={reqs}")

for name, B, L, w in shapes:
    sim_plan(name, B, L, w)
