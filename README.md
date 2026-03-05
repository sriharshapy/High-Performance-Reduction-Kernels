# High-Performance Reduction Kernels

High-throughput 1D sum-reduction benchmarks across four backends — hand-written CUDA C, NVIDIA CUB, Triton, and a PyTorch baseline — all sharing a common CLI and verification scheme for apples-to-apples comparison.

---

## Layout

| File | Description |
|---|---|
| `reductions/sum/vector_sum_reduction.cu` | Hand-tuned CUDA C kernel |
| `reductions/sum/vector_sum_reduction_cub.cu` | CUDA + CUB `DeviceReduce::Sum` |
| `reductions/sum/vector_sum_reduction.py` | Triton two-pass reduction |
| `reductions/sum/vector_sum_reduction_torch.py` | PyTorch `torch.sum` baseline |
| `reductions/sum/input_gen_1D.py` | Binary float32 input generator |
| `reductions/sum/reduction_sum.ipynb` | **Main benchmark orchestration + analysis notebook** |

All kernels read a raw float32 binary file or fall back to synthetic data `x[i] = 1/(i+1)`.

---

## Requirements

- **CUDA-capable GPU** + recent NVIDIA driver
- **CUDA toolkit** (to compile `.cu` files)
- **Python 3** with `torch`, `triton`, `numpy`, `pandas`, `matplotlib`, `jupyter`

---

## Quickstart

**1. Generate input data**
```bash
python reductions/sum/input_gen_1D.py out.bin -n 1000000000 --seed 42
```

**2. Compile the CUDA kernels** (auto-detect arch)
```bash
cd reductions/sum
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
nvcc -O3 -arch=sm_${ARCH} -use_fast_math -o vector_sum_reduction vector_sum_reduction.cu
nvcc -O3 -arch=sm_${ARCH} -use_fast_math -o vector_sum_reduction_cub vector_sum_reduction_cub.cu
```

**3. Run the full benchmark + analysis via the notebook**

The notebook in `reductions/sum/reduction_sum.ipynb` orchestrates everything: it compiles the CUDA binaries, sweeps over multiple values of `n`, runs all four implementations, collects timings, and saves charts + tables to `analysis_output/`.

```bash
# Locally
jupyter notebook reductions/sum/reduction_sum.ipynb

# Or on Google Colab — mount your Drive and open the notebook directly
```

Run all cells top-to-bottom. Outputs are saved to `analysis_output/`.

---

## Results (NVIDIA T4, sm_75)

All implementations pass correctness verification at every size. Timings are averaged over 100 kernel launches.

### Fastest implementation at each `n`

```
            n   fastest   ms
       10,000   CUDA C    0.0104
      100,000   CUDA C    0.0079
      500,000   CUDA C    0.0132
    1,000,000   CUDA C    0.0137
   10,000,000   CUDA C    0.1584
   50,000,000   CUDA C    0.7416
  100,000,000   CUDA C    1.4716
  500,000,000  PyTorch    7.2570
1,000,000,000  PyTorch   14.4983
```

### Why hand-tuned CUDA C wins at small-to-medium sizes

The hand-written kernel (`vector_sum_reduction.cu`) outperforms all others from 10K to 100M elements — by a wide margin at small sizes:

| n | CUDA C (ms) | CUB (ms) | Triton (ms) | PyTorch (ms) | CUDA C vs PyTorch | CUDA C vs Triton |
|---|---|---|---|---|---|---|
| 10K | **0.0104** | 0.0128 | 0.0642 | 0.0532 | **~5× faster** | **~6× faster** |
| 100K | **0.0079** | 0.0099 | 0.0521 | 0.0478 | **~6× faster** | **~6.6× faster** |
| 1M | **0.0137** | 0.0160 | 0.0544 | 0.0359 | **~2.6× faster** | **~4× faster** |
| 10M | **0.1584** | 0.1597 | 0.1938 | 0.1722 | ~9% faster | ~22% faster |
| 100M | **1.4716** | 1.4971 | 1.5339 | 1.4822 | ~0.7% faster | ~4% faster |

**Key design decisions that drive the wins:**

- **`float4` vectorized loads** — reads 4 floats per instruction, maximizing memory bus utilization and reducing load instruction count.
- **Warp-shuffle reduction** (`__shfl_down_sync`) — no shared memory needed for the intra-warp pass, eliminating bank conflicts entirely.
- **Minimal shared memory** — only one `float` per warp is staged in shared memory for the cross-warp reduction (32 floats total), keeping the footprint tiny.
- **SM-aware launch config** — block count is computed from `multiProcessorCount × max_blocks_per_SM` to saturate all SMs without over-subscribing the second-pass reduction.
- **Contiguous block assignment** — each block is assigned a contiguous chunk of the input rather than a global stride pattern, improving L1/L2 cache locality.

### Why PyTorch wins at very large sizes (≥500M)

At 500M–1B elements all kernels become fully **DRAM bandwidth-bound** — the GPU is spending nearly all its time waiting on memory. In that regime, PyTorch's highly tuned internal kernel (itself built on CUB under the hood) edges ahead slightly, while the hand-written kernel's extra occupancy optimizations provide no additional benefit.

---

## License

MIT License — see [LICENSE](LICENSE).
