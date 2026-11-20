with open('submission.py', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        low = line.lower()
        if 'moe' in low and ('run' in low or 'ext' in low or 'dispatch' in low or 'prepare' in low):
            if any(k in line for k in ['def ', 'okc_run', 'ext(', 'load_inline', 'st[']):
                print(f"{i+1}: {line.strip()[:100]}")
