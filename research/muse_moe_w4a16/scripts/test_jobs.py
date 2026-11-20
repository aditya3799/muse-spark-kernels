import random

def test_jobs():
    # For M=32, total local tokens is ~50.
    # If FT_T = 32:
    # Any expert with > 8 tokens is a fat job of size <= 32.
    # Since max tokens on an expert for M=32 is ~14 (< 16),
    # with FT_T = 32, every fat expert becomes 1 fat job of size <= 14 tokens (ntt <= 2)!
    # With FT_T = 16:
    # Every fat expert also becomes 1 fat job of size <= 14 tokens (ntt <= 2)!
    # Because NO expert has > 16 tokens!
    print("Under M=32, since max tokens is <= 16, both FT_T=32 and FT_T=16 produce THE EXACT SAME NUMBER OF JOBS (1 job per expert)!")
    print("BUT with FT_T=16, WGS_NT=2 instead of 4:")
    print("  - Shared memory per slot WGS_XS drops from 16 KB to 8 KB (saving 32 KB smem!)")
    print("  - Accumulator registers d[WGS_NT][4] drops from 16 registers to 8 registers (saving 8 VGPRs per thread across all 512 threads!)")
    print("  - WGS_FIXW drops from 2048 floats to 1024 floats!")

test_jobs()
