import math

# For M=32:
# Average active local experts is ~16.
# Each active local expert has 1 job.
# In Phase A:
# Each job has 32 units (256 rows each, total 8192 rows for gate+up).
# Total units in Phase A = 16 experts * 32 units = 512 units!
# In Phase B:
# Each job has 16 units (256 rows each, total 4096 rows for down).
# Total units in Phase B = 16 experts * 16 units = 256 units!
# Total units across both phases = 768 units.
# With 132 CTAs:
# 768 / 132 = ~5.8 units per CTA!
print("Phase A units:", 16 * 32, "-> per CTA (132 SMs):", 512 / 132)
print("Phase B units:", 16 * 16, "-> per CTA (132 SMs):", 256 / 132)
print("Every CTA processes ~3.8 units in Phase A and ~1.9 units in Phase B.")
