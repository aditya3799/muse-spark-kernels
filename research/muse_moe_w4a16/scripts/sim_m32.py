import random
import math

def simulate_m32(n_trials=10000):
    M = 32
    max_tokens_list = []
    over_16_count = 0
    for _ in range(n_trials):
        # generate random routing for M=32
        # top 8 out of 256 experts
        # local experts are 32 experts (e.g. 0..31)
        # In the benchmark, topk experts are selected based on softmax/scores
        expert_counts = [0] * 32
        for t in range(M):
            # pick 8 unique experts out of 256
            experts = random.sample(range(256), 8)
            for e in experts:
                if e < 32:
                    expert_counts[e] += 1
        m = max(expert_counts)
        max_tokens_list.append(m)
        if m > 16:
            over_16_count += 1
    print(f"M=32 over {n_trials} runs:")
    print(f"Max tokens seen across all runs: {max(max_tokens_list)}")
    print(f"Runs with max > 16: {over_16_count} / {n_trials} ({over_16_count/n_trials*100:.2f}%)")
    print(f"Runs with max > 24: {sum(1 for x in max_tokens_list if x > 24)} / {n_trials}")
    print(f"Avg max tokens: {sum(max_tokens_list)/len(max_tokens_list):.2f}")

simulate_m32()
