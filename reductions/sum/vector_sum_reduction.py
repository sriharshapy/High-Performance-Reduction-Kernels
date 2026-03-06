"""
Fast 1D vector sum reduction (Triton).

Two-pass pattern: each program sums blocks of input (grid-stride if needed),
writes one partial; second kernel sums partials to a single scalar.
Uses power-of-2 block sizes and device-dependent program count for occupancy.
Mirrors the CUDA vector_sum_reduction.cu setup: warmup, timed runs with
CUDA events, verification.

Run: python vector_sum_reduction.py [input_file] [n]
     input_file: .npy or raw binary float32 (default: out.bin)
     n: number of elements to reduce, i.e. first n from file (default: 1000000000)
     Uses synthetic data 1/(i+1) if file not found.
"""

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Pass 1: each program sums one or more blocks (grid-stride), writes one partial.
# BLOCK_SIZE must be power of 2 for coalescing.
# -----------------------------------------------------------------------------
@triton.jit
def reduce_sum_kernel_forward(
    x_ptr,
    partial_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_programs,
    num_steps,
):
    pid = tl.program_id(axis=0)
    acc = tl.zeros((1,), dtype=tl.float32)
    for step in tl.range(0, num_steps):
        block_idx = pid + step * num_programs
        if block_idx * BLOCK_SIZE < n_elements:
            start = block_idx * BLOCK_SIZE
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            block = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            acc += tl.sum(block, axis=0)
    tl.store(partial_ptr + pid, tl.sum(acc, axis=0))


# -----------------------------------------------------------------------------
# Pass 2: one program sums all partials (up to BLOCK_SIZE_2 elements).
# -----------------------------------------------------------------------------
@triton.jit
def reduce_sum_kernel_final(
    partial_ptr,
    out_ptr,
    n_partials,
    BLOCK_SIZE_2: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE_2)
    mask = offsets < n_partials
    block = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    s = tl.sum(block, axis=0)
    tl.store(out_ptr, s)


def get_reduction_config(n: int, block_size: int = 1024, max_partials: int = 1024,
                         num_warps: int = 4):
    """Choose num_programs and num_steps for occupancy (device-aware).

    num_warps must match the num_warps used at kernel launch (Triton default: 4).
    threads_per_program = num_warps * warp_size (32), so the default is 128, not 256.
    Using 256 would halve max_blocks_per_sm and under-schedule small-to-medium arrays.
    """
    total_blocks = (n + block_size - 1) // block_size
    try:
        props = torch.cuda.get_device_properties(0)
        num_sms = props.multi_processor_count
        max_threads_per_sm = getattr(props, "max_threads_per_multiprocessor", 1024)
        threads_per_program = num_warps * 32  # warp size is always 32 on NVIDIA GPUs
        max_blocks_per_sm = max(1, max_threads_per_sm // threads_per_program)
        target_programs = num_sms * max_blocks_per_sm
    except Exception:
        target_programs = 256
    num_programs = min(max_partials, max(total_blocks, target_programs))
    num_programs = max(1, num_programs)
    num_steps = (total_blocks + num_programs - 1) // num_programs
    return num_programs, num_steps, total_blocks


_NUM_WARPS = 4  # Triton default; must stay in sync with get_reduction_config call below


def reduce_sum_triton(x: torch.Tensor, block_size: int = 1024, max_partials: int = 1024):
    """Two-pass sum reduction; returns scalar tensor on same device as x."""
    n = x.numel()
    if n == 0:
        return x.sum()
    num_programs, num_steps, _ = get_reduction_config(n, block_size, max_partials,
                                                       num_warps=_NUM_WARPS)
    partial = torch.empty((num_programs,), device=x.device, dtype=torch.float32)
    out = torch.empty((1,), device=x.device, dtype=torch.float32)
    BLOCK_SIZE_2 = triton.next_power_of_2(num_programs)
    reduce_sum_kernel_forward[(num_programs,)](
        x,
        partial,
        n_elements=n,
        BLOCK_SIZE=block_size,
        num_programs=num_programs,
        num_steps=num_steps,
        num_warps=_NUM_WARPS,
    )
    reduce_sum_kernel_final[(1,)](
        partial,
        out,
        n_partials=num_programs,
        BLOCK_SIZE_2=BLOCK_SIZE_2,
        num_warps=_NUM_WARPS,
    )
    return out[0]


def main():
    parser = argparse.ArgumentParser(
        description="Triton 1D vector sum reduction",
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

    import numpy as np

    # Load first n elements from file, or generate synthetic data
    path_obj = Path(path) if path else None
    if path_obj and path_obj.exists() and path_obj.is_file():
        if path_obj.suffix.lower() == ".npy":
            arr = np.load(path_obj).astype(np.float32).ravel()
            if arr.size < n:
                print(f"Error: file has {arr.size} elements, need at least {n} (first n)", file=sys.stderr)
                return 1
            arr = arr[:n].copy()
        else:
            try:
                arr = np.fromfile(path_obj, dtype=np.float32, count=n)
            except Exception as e:
                print(f"Failed to read binary file: {e}", file=sys.stderr)
                return 1
            if arr.size < n:
                print(f"Error: file has {arr.size} elements, need {n} (first n elements)", file=sys.stderr)
                return 1
    else:
        arr = np.array([1.0 / (i + 1) for i in range(n)], dtype=np.float32)

    x = torch.from_numpy(arr).cuda()
    n = x.numel()

    block_size = 1024
    max_partials = 1024
    num_programs, num_steps, _ = get_reduction_config(n, block_size, max_partials,
                                                       num_warps=_NUM_WARPS)

    # Partial and output buffers
    partial = torch.empty((num_programs,), device=x.device, dtype=torch.float32)
    out = torch.empty((1,), device=x.device, dtype=torch.float32)
    BLOCK_SIZE_2 = triton.next_power_of_2(num_programs)

    if do_warmup:
        warmup_iters = 3
        for _ in range(warmup_iters):
            reduce_sum_kernel_forward[(num_programs,)](
                x, partial, n_elements=n, BLOCK_SIZE=block_size,
                num_programs=num_programs, num_steps=num_steps,
                num_warps=_NUM_WARPS,
            )
            reduce_sum_kernel_final[(1,)](
                partial, out, n_partials=num_programs, BLOCK_SIZE_2=BLOCK_SIZE_2,
                num_warps=_NUM_WARPS,
            )
        torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(timed_iters):
        start_ev.record()
        reduce_sum_kernel_forward[(num_programs,)](
            x, partial, n_elements=n, BLOCK_SIZE=block_size,
            num_programs=num_programs, num_steps=num_steps,
            num_warps=_NUM_WARPS,
        )
        reduce_sum_kernel_final[(1,)](
            partial, out, n_partials=num_programs, BLOCK_SIZE_2=BLOCK_SIZE_2,
            num_warps=_NUM_WARPS,
        )
        end_ev.record()
        torch.cuda.synchronize()
        total_ms += start_ev.elapsed_time(end_ev)
    avg_ms = total_ms / timed_iters
    sum_val = out[0].item()

    try:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        num_sms = 0
    print(f"n = {n}")
    print(f"num_programs = {num_programs}")
    print(f"block_size = {block_size}")
    print(f"SMs = {num_sms}")
    print(f"sum = {sum_val:.10g}")
    print(f"avg_kernel_ms = {avg_ms:.4f}")
    print(f"niterations = {timed_iters}")

    # Reference: sequential float sum on CPU (match CUDA reference)
    ref = float(x.cpu().float().sum().item())
    err = abs(sum_val - ref)
    tol = 1e-3 * (abs(ref) + 1.0)
    ok = err <= tol
    print(f"err = {err:.6g}")
    print(f"tol = {tol:.6g}")
    if ok:
        print("VERIFY PASS")
    else:
        print(f"VERIFY FAIL: triton={sum_val:.10g} ref={ref:.10g}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
