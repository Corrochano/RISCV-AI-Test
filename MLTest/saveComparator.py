import re
import os
import sys


def parse_time_to_seconds(t):
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

    # Extract base filename (e.g., bertTime.txt)
    base_name = os.path.basename(file1)
    base_name = os.path.splitext(base_name)[0]

    output_file = f"{base_name}_comparison.txt"

    with open(output_file, "w") as out:

        out.write("COMMON OPERATIONS\n")
        out.write(f"{'Operation'.ljust(45)} File1(s) File2(s) Diff(s)\n")
        out.write("-" * 70 + "\n")

        for op in common:
            t1 = p1[op]
            t2 = p2[op]
            diff = t2 - t1
            out.write(f"{op.ljust(45)} {t1:10.6f} {t2:10.6f} {diff:10.6f}\n")

        out.write("\nONLY IN FILE1\n")
        for op in only1:
            out.write(op + "\n")

        out.write("\nONLY IN FILE2\n")
        for op in only2:
            out.write(op + "\n")

    print(f"Results written to: {output_file}")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python comparator.py profile1.txt profile2.txt")
        sys.exit(1)

    compare_profiles(sys.argv[1], sys.argv[2])
