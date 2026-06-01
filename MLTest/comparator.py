import re

def parse_time_to_seconds(t):
    """Convert PyTorch profiler time string to seconds."""
    t = t.strip()

    if t.endswith("ms"):
        return float(t[:-2]) / 1000
    elif t.endswith("us"):
        return float(t[:-2]) / 1e6
    elif t.endswith("s"):
        return float(t[:-1])
    else:
        return 0.0


def read_profile(path):
    ops = {}

    with open(path) as f:
        for line in f:
            line = line.rstrip()

            if "aten::" not in line:
                continue

            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) < 3:
                continue

            name = parts[0]
            self_cpu = parts[2]

            ops[name] = parse_time_to_seconds(self_cpu)

    return ops


def compare_profiles(file1, file2):
    p1 = read_profile(file1)
    p2 = read_profile(file2)

    common = sorted(set(p1) & set(p2))
    only1 = sorted(set(p1) - set(p2))
    only2 = sorted(set(p2) - set(p1))

    print("\nCOMMON OPERATIONS")
    print("Operation".ljust(45), "File1(s)", "File2(s)", "Diff(s)")
    print("-" * 70)

    for op in common:
        t1 = p1[op]
        t2 = p2[op]
        diff = t2 - t1
        print(f"{op.ljust(45)} {t1:10.6f} {t2:10.6f} {diff:10.6f}")

    print("\nONLY IN FILE1")
    for op in only1:
        print(op)

    print("\nONLY IN FILE2")
    for op in only2:
        print(op)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python comparator.py profile1.txt profile2.txt")
        exit(1)

    compare_profiles(sys.argv[1], sys.argv[2])
