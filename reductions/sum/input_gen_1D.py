"""
Input file generator for the sum-reduction benchmark.

Generates a 1D array of random floats in the range [-1, 1], stored as binary
float32 (little-endian, no header). Used to feed CUDA C, Triton, or other
implementations of vector sum reduction for reproducible benchmarking.

Output format:
    Raw binary: N consecutive float32 values (4 bytes each). Total file size
    in bytes = N * 4. Read with the same dtype and known N (or file size / 4).

Usage:
    python input_gen.py <output> [-n SIZE] [--seed SEED]

Examples:
    python input_gen.py data.bin
    python input_gen.py data.bin -n 1000000 --seed 42

Reading the file:
    Python:  x = np.fromfile("data.bin", dtype=np.float32)
    C/CUDA:  read as float[] or float* with length = file_size / sizeof(float)
"""
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Generate 1D random float input for reduction (sum).")
    p.add_argument("output", help="Output file path")
    p.add_argument("-n", "--size", type=int, default=1_000_000, help="Number of elements (default: 1e6)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = p.parse_args()

    # Uniform [-1, 1], float32 for consistency with typical GPU kernels
    rng = np.random.default_rng(args.seed)
    x = rng.uniform(low=-1.0, high=1.0, size=args.size).astype(np.float32)
    x.tofile(args.output)
    print(f"Wrote {args.size} float32 values in [-1, 1] to {args.output}")


if __name__ == "__main__":
    main()
