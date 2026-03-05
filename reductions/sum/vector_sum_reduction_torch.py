"""
Fast 1D vector sum reduction (PyTorch).

Baseline implementation using `torch.sum` on a CUDA tensor.
Mirrors the CLI, timing, and verification pattern of the Triton and CUDA
implementations in this directory.

Run: python vector_sum_reduction_torch.py [--no-warmup] [input_file] [n] [timed_iters]
     input_file: .npy or raw binary float32 (default: out.bin)
     n         : number of elements to reduce, i.e. first n from file (default: 1_000_000_000)
     timed_iters: number of timed iterations for averaging (default: 100)
     If the file is missing, uses synthetic data 1/(i+1) for n elements.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def load_input(path: str, n: int) -> np.ndarray:
    """Load first n float32 elements from path or generate synthetic data."""
    if n <= 0:
        raise ValueError("n must be positive")

    path_obj = Path(path) if path else None
    if path_obj and path_obj.exists() and path_obj.is_file():
        if path_obj.suffix.lower() == ".npy":
            arr = np.load(path_obj).astype(np.float32).ravel()
            if arr.size < n:
                raise ValueError(f"file has {arr.size} elements, need at least {n} (first n)")
            arr = arr[:n].copy()
        else:
            try:
                arr = np.fromfile(path_obj, dtype=np.float32, count=n)
            except Exception as e:
                raise RuntimeError(f"Failed to read binary file: {e}") from e
            if arr.size < n:
                raise ValueError(f"file has {arr.size} elements, need {n} (first n elements)")
    else:
        # Synthetic data to match other implementations: 1/(i+1)
        arr = np.array([1.0 / (i + 1) for i in range(n)], dtype=np.float32)
    return arr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PyTorch 1D vector sum reduction (torch.sum on CUDA)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable warmup runs before timed iterations",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="out.bin",
        help="Input file: .npy or raw binary float32 (default: out.bin)",
    )
    parser.add_argument(
        "n",
        nargs="?",
        type=int,
        default=1_000_000_000,
        help="Number of elements to reduce, i.e. first n from file (default: 1e9)",
    )
    parser.add_argument(
        "timed_iters",
        nargs="?",
        type=int,
        default=100,
        help="Number of timed iterations for averaging (default: 100)",
    )
    args = parser.parse_args()

    path = args.input
    n = args.n
    timed_iters = args.timed_iters
    do_warmup = not args.no_warmup

    if n <= 0:
        print("Error: n must be positive", file=sys.stderr)
        return 1
    if timed_iters <= 0:
        print("Error: timed_iters must be positive", file=sys.stderr)
        return 1

    try:
        arr = load_input(path, n)
    except Exception as e:
        print(f"Error loading input: {e}", file=sys.stderr)
        return 1

    # Move to GPU
    x = torch.from_numpy(arr).cuda()
    n = x.numel()

    if do_warmup:
        warmup_iters = 3
        for _ in range(warmup_iters):
            y = x.sum()
            # Avoid D2H sync cost in warmup; just keep it on device
            _ = y  # noqa: F841
        torch.cuda.synchronize()

    # Timing with CUDA events, matching other scripts
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    sum_val = None
    for _ in range(timed_iters):
        start_ev.record()
        y = x.sum()
        end_ev.record()
        torch.cuda.synchronize()
        total_ms += start_ev.elapsed_time(end_ev)
        # Grab scalar once per iteration so we don't measure host sync in timing
        sum_val = float(y.item())

    avg_ms = total_ms / timed_iters

    try:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        num_sms = 0

    print(f"n = {n}")
    print(f"SMs = {num_sms}")
    print(f"sum = {sum_val:.10g}")
    print(f"avg_kernel_ms = {avg_ms:.4f}")
    print(f"niterations = {timed_iters}")

    # Reference: sequential float sum on CPU, matching Triton script's approach
    ref = float(x.cpu().float().sum().item())
    err = abs(sum_val - ref)
    tol = 1e-3 * (abs(ref) + 1.0)
    ok = err <= tol
    print(f"err = {err:.6g}")
    print(f"tol = {tol:.6g}")
    if ok:
        print("VERIFY PASS")
    else:
        print(f"VERIFY FAIL: torch={sum_val:.10g} ref={ref:.10g}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

