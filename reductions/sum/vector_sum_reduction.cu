/*
 * Fast 1D vector sum reduction (CUDA).
 *
 * Pattern: strided per-thread accumulation → warp shuffle reduction
 * → shared memory for block partial sums → second pass to single scalar.
 * Based on:
 *   - Ash Vardanian, ParallelReductionsBenchmark (reduce_cuda.cuh)
 *     https://github.com/ashvardanian/ParallelReductionsBenchmark
 *   - NVIDIA "Faster Parallel Reductions on Kepler" / warp-level primitives
 *     https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
 *
 * Accumulation loop unrolling (REDUCE_UNROLL):
 *   A single-accumulator loop creates a serial dependency chain: each iteration
 *   must wait for the previous `sum +=` to complete before the address for the
 *   next load can be resolved.  The GPU's memory subsystem can service many
 *   in-flight loads simultaneously, but a single accumulator never keeps more
 *   than one outstanding, leaving the ~200-cycle DRAM latency fully exposed.
 *   Using REDUCE_UNROLL independent accumulators breaks the chain: all
 *   REDUCE_UNROLL loads are issued before any result is needed, keeping the
 *   load pipeline full.  The partial sums are folded into one scalar after the
 *   loop.  This mirrors what CUB does internally with 4–8 accumulators.
 *
 * Dynamic access-pattern selection (CONTIGUOUS template parameter):
 *   Contiguous block assignment creates long sequential streams that the GPU
 *   hardware prefetcher can exploit, keeping cache lines arriving well beyond
 *   the physical L2 boundary.  Empirically on sm_75 (T4, 4 MB L2) the
 *   contiguous pattern wins up to ~100 M floats (400 MB = 100 × L2); beyond
 *   that the grid-stride pattern wins by maximising DRAM channel parallelism.
 *   The threshold is set to 100 × cudaDevAttrL2CacheSize so it scales
 *   automatically with the device (T4 → 400 MB, A100 → 4 GB, H100 → 5 GB).
 *
 * Compile: nvcc -O3 -arch=sm_80 -o vector_sum_reduction vector_sum_reduction.cu
 * Run:     ./vector_sum_reduction [input_file] [n]
 *          input_file : raw binary float32 (default: out.bin)
 *          n          : number of elements to reduce, i.e. first n from file (default: 1000000000)
 *          If no file or file missing, uses synthetic data 1/(i+1) for n elements.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define WARP_SIZE    32
#define FULL_MASK    0xffffffffu
// Number of independent accumulators in the vectorized load loop.
// Each accumulator has no data dependency on the others, so the compiler
// can issue all REDUCE_UNROLL loads before any result arrives, saturating
// the memory pipeline and hiding DRAM latency.
#define REDUCE_UNROLL 4

// -----------------------------------------------------------------------------
// Warp-level sum reduction using shuffle (no shared memory; avoids bank conflicts).
// All 32 threads in the warp participate; lane 0 holds the sum.
// -----------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  return val;
}


// -----------------------------------------------------------------------------
// Block-level sum: each thread has a partial sum; reduce to one value per block.
// Uses warp shuffle then shared memory to combine warps (supports any block size
// that is a multiple of WARP_SIZE).
// -----------------------------------------------------------------------------
__device__ float block_reduce_sum(float val) {
  __shared__ float shared[32];  // max warps per block typically 32
  int lane = threadIdx.x % WARP_SIZE;
  int wid  = threadIdx.x / WARP_SIZE;

  val = warp_reduce_sum(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  // First warp reads partial sums from shared (pad with 0 if fewer warps)
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.f;
  if (wid == 0) val = warp_reduce_sum(val);
  return val;
}

// -----------------------------------------------------------------------------
// Kernel: REDUCE_UNROLL independent accumulators hide DRAM latency (see header).
//
// Template parameter CONTIGUOUS selects the access pattern:
//   true  — each block owns a contiguous segment of float4 chunks; improves
//            L1/L2 reuse when the whole dataset fits (or partially fits) in cache.
//   false — global grid-stride loop; all blocks issue requests to interleaved
//            DRAM channels simultaneously when data >> L2 (no locality benefit).
//
// The host selects the pattern at launch time based on n vs L2 cache size
// (see use_contiguous_blocks()), so no runtime branch lives in the hot loop.
// -----------------------------------------------------------------------------
template <bool CONTIGUOUS>
__global__ void reduce_sum_kernel_float_in(const float* __restrict__ inputs,
                                           size_t n,
                                           float* __restrict__ partial_sums) {
  const int  threads_per_block = blockDim.x;
  const int  num_blocks        = gridDim.x;
  const int  tid               = threadIdx.x;
  const int  bid               = blockIdx.x;

  const size_t vec_n4        = n / 4;
  const size_t stride        = (size_t)threads_per_block;
  const size_t total_threads = stride * (size_t)num_blocks;
  const size_t global_tid    = (size_t)bid * stride + (size_t)tid;

  const float4* inputs4 = reinterpret_cast<const float4*>(inputs);

  // Four independent accumulators — no cross-accumulator data dependencies.
  float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;

  if constexpr (CONTIGUOUS) {
    // Each block covers a contiguous segment; threads stride within it.
    // Good when data fits (or partially fits) in L2: subsequent iterations
    // hit cache-warm lines before they are evicted.
    const size_t float4_per_block = (vec_n4 + (size_t)num_blocks - 1) / (size_t)num_blocks;
    const size_t start4 = (size_t)bid * float4_per_block;
    size_t       end4   = start4 + float4_per_block;
    if (end4 > vec_n4) end4 = vec_n4;

    size_t idx4 = start4 + (size_t)tid;
    for (; idx4 + (REDUCE_UNROLL - 1) * stride < end4; idx4 += REDUCE_UNROLL * stride) {
      const float4 v0 = inputs4[idx4];
      const float4 v1 = inputs4[idx4 +     stride];
      const float4 v2 = inputs4[idx4 + 2 * stride];
      const float4 v3 = inputs4[idx4 + 3 * stride];
      sum0 += v0.x + v0.y + v0.z + v0.w;
      sum1 += v1.x + v1.y + v1.z + v1.w;
      sum2 += v2.x + v2.y + v2.z + v2.w;
      sum3 += v3.x + v3.y + v3.z + v3.w;
    }
#pragma unroll 1
    for (; idx4 < end4; idx4 += stride) {
      const float4 v = inputs4[idx4];
      sum0 += v.x + v.y + v.z + v.w;
    }
  } else {
    // Global grid-stride loop: adjacent threads in a warp access adjacent
    // float4 chunks (coalesced), and all blocks interleave across the full
    // address space, hitting every DRAM channel simultaneously.
    // Preferred when data >> L2 (no locality to exploit).
    size_t idx4 = global_tid;
    for (; idx4 + (REDUCE_UNROLL - 1) * total_threads < vec_n4;
           idx4 += REDUCE_UNROLL * total_threads) {
      const float4 v0 = inputs4[idx4];
      const float4 v1 = inputs4[idx4 +     total_threads];
      const float4 v2 = inputs4[idx4 + 2 * total_threads];
      const float4 v3 = inputs4[idx4 + 3 * total_threads];
      sum0 += v0.x + v0.y + v0.z + v0.w;
      sum1 += v1.x + v1.y + v1.z + v1.w;
      sum2 += v2.x + v2.y + v2.z + v2.w;
      sum3 += v3.x + v3.y + v3.z + v3.w;
    }
#pragma unroll 1
    for (; idx4 < vec_n4; idx4 += total_threads) {
      const float4 v = inputs4[idx4];
      sum0 += v.x + v.y + v.z + v.w;
    }
  }

  // Fold the four independent partial sums.
  float sum = (sum0 + sum1) + (sum2 + sum3);

  // Scalar tail for remaining elements (n % 4).
  const size_t tail_start = vec_n4 * 4;
  for (size_t i = tail_start + global_tid; i < n; i += total_threads)
    sum += inputs[i];

  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) partial_sums[blockIdx.x] = sum;
}

// -----------------------------------------------------------------------------
// Second pass: reduce partial_sums[0..n-1] to partial_sums[0].
// Uses one thread per partial (rounded up to warp) for fully coalesced reads;
// avoids strided access and excess DRAM traffic vs. fixed 256-thread stride loop.
// -----------------------------------------------------------------------------
__global__ void reduce_sum_kernel_float_in_pass2(const float* __restrict__ inputs,
                                                 size_t n,
                                                 float* __restrict__ partial_sums) {
  size_t i = (size_t)threadIdx.x;
  float sum = (i < n) ? inputs[i] : 0.0f;

  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) partial_sums[0] = sum;
}

// -----------------------------------------------------------------------------
// Return true when contiguous block assignment is expected to be faster.
//
// Why the threshold is 100 × L2, not 1 × L2:
//   The GPU hardware prefetcher recognises long sequential streams and issues
//   speculative loads ahead of the program counter.  Contiguous block
//   assignment creates exactly those streams (each block walks a single
//   contiguous region), so the prefetcher can keep cache lines arriving even
//   when the working set far exceeds physical L2.  Benchmarks on sm_75 (T4,
//   4 MB L2) show contiguous wins up to ~100 M floats (400 MB = 100 × L2):
//
//     n            fastest       ms
//     10 000       CUDA C        0.0090
//     100 000      CUDA C        0.0107
//     500 000      CUDA C CUB    0.0138   ← CUB barely edges ahead here
//     1 000 000    CUDA C        0.0134
//     10 000 000   CUDA C CUB    0.1596   ← very close
//     50 000 000   CUDA C        0.7514
//     100 000 000  CUDA C        1.4834   ← last win for contiguous
//     500 000 000  PyTorch       7.2716   ← grid-stride takes over
//
//   Once data >> 100 × L2 the prefetcher provides no further benefit and the
//   grid-stride pattern wins because all blocks interleave DRAM channel
//   requests simultaneously instead of serialising over their own segment.
//
// Scaling by 100 × L2 keeps the threshold proportional across generations:
//   T4   (sm_75, 4 MB L2)  →  400 MB  =  100 M floats
//   A100 (sm_80, 40 MB L2) →  4 GB    =    1 B floats
//   H100 (sm_90, 50 MB L2) →  5 GB    =  1.25 B floats
//
// Fallback when L2 size cannot be queried: use the empirical 400 MB value.
// -----------------------------------------------------------------------------
static bool use_contiguous_blocks(size_t n) {
  int l2_bytes = 0;
  cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0);
  const size_t threshold = (l2_bytes > 0)
      ? (size_t)l2_bytes * 100          // device-proportional: 100 × L2
      : (size_t)400 * 1024 * 1024;      // fallback: 400 MB (empirical T4 value)
  return n * sizeof(float) <= threshold;
}

// -----------------------------------------------------------------------------
// Host: run two-pass reduction in float, return result.
// Second pass uses one block with one thread per partial (coalesced) to cut DRAM.
// -----------------------------------------------------------------------------
static float reduce_sum(const float* d_inputs, size_t n, int blocks, int threads_per_block,
                        float* d_partial, bool contiguous) {
  if (contiguous)
    reduce_sum_kernel_float_in<true><<<blocks, threads_per_block>>>(d_inputs, n, d_partial);
  else
    reduce_sum_kernel_float_in<false><<<blocks, threads_per_block>>>(d_inputs, n, d_partial);

  // One thread per partial, round up to warp size; cap at 1024
  int pass2_threads = (int)((blocks + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
  if (pass2_threads > 1024) pass2_threads = 1024;
  if (pass2_threads < WARP_SIZE) pass2_threads = WARP_SIZE;

  reduce_sum_kernel_float_in_pass2<<<1, pass2_threads>>>(d_partial, (size_t)blocks, d_partial);
  cudaDeviceSynchronize();
  float result;
  cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
  return result;
}

// -----------------------------------------------------------------------------
// Choose block count for occupancy: at least (SMs * blocks_per_SM), cap for second pass.
// -----------------------------------------------------------------------------
static void get_reduction_launch_config(size_t n, int threads_per_block,
                                        int* out_blocks, int* out_max_blocks) {
  int num_sms = 0;
  int max_threads_per_sm = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

  int max_blocks_per_sm = max_threads_per_sm / threads_per_block;
  if (max_blocks_per_sm < 1) max_blocks_per_sm = 1;
  int target_blocks = num_sms * max_blocks_per_sm;  // minimum to fill all SMs
  int max_blocks = target_blocks;
  if (max_blocks > 1024) max_blocks = 1024;  // cap so second-pass reduction stays small
  if (max_blocks < 1) max_blocks = 1;

  size_t blocks_needed = (n + (size_t)threads_per_block - 1) / (size_t)threads_per_block;
  int blocks = (int)blocks_needed;
  if (blocks > max_blocks) blocks = max_blocks;
  if (blocks < 1) blocks = 1;

  *out_blocks = blocks;
  *out_max_blocks = max_blocks;
}

// -----------------------------------------------------------------------------
// Main: [--no-warmup] input_file [n] [timed_iters]; use first n elements.
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
  const char* path = "out.bin";
  size_t n = 1000000000;
  int timed_iters = 100;
  int do_warmup = 1;

  int a = 1;
  if (a < argc && strcmp(argv[a], "--no-warmup") == 0) {
    do_warmup = 0;
    a++;
  }
  if (a < argc) path = argv[a++];
  if (a < argc) {
    long long n_arg = atoll(argv[a++]);
    if (n_arg <= 0) { fprintf(stderr, "Error: n must be positive (got %lld)\n", (long long)n_arg); return 1; }
    n = (size_t)n_arg;
  }
  if (a < argc) {
    int ti = atoi(argv[a++]);
    if (ti <= 0) { fprintf(stderr, "Error: timed_iters must be positive (got %d)\n", ti); return 1; }
    timed_iters = ti;
  }

  float* h_input = (float*)malloc(n * sizeof(float));
  if (!h_input) { fprintf(stderr, "malloc failed\n"); return 1; }

  {
    FILE* f = fopen(path, "rb");
    if (f) {
      size_t read = fread(h_input, sizeof(float), n, f);
      fclose(f);
      if (read != n) {
        fprintf(stderr, "Error: file has %zu floats, need %zu (first n elements)\n", read, n);
        free(h_input);
        return 1;
      }
    } else {
      for (size_t i = 0; i < n; i++) h_input[i] = 1.f / (float)(i + 1);
    }
  }

  int threads = 256;
  int blocks = 0, max_blocks = 0;
  get_reduction_launch_config(n, threads, &blocks, &max_blocks);

  // Decide access pattern once, before any kernel launch.
  const bool contiguous = use_contiguous_blocks(n);

  float* d_input   = nullptr;
  float* d_partial = nullptr;
  cudaMalloc(&d_input, n * sizeof(float));
  cudaMalloc(&d_partial, (size_t)max_blocks * sizeof(float));

  cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

  if (do_warmup) {
    const int warmup_iters = 3;
    for (int w = 0; w < warmup_iters; w++)
      (void)reduce_sum(d_input, n, blocks, threads, d_partial, contiguous);
  }

  int pass2_threads = (int)((blocks + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
  if (pass2_threads > 1024) pass2_threads = 1024;
  if (pass2_threads < WARP_SIZE) pass2_threads = WARP_SIZE;

  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);
  float total_ms = 0.f;
  for (int t = 0; t < timed_iters; t++) {
    cudaEventRecord(ev_start);
    if (contiguous)
      reduce_sum_kernel_float_in<true><<<blocks, threads>>>(d_input, n, d_partial);
    else
      reduce_sum_kernel_float_in<false><<<blocks, threads>>>(d_input, n, d_partial);
    reduce_sum_kernel_float_in_pass2<<<1, pass2_threads>>>(d_partial, (size_t)blocks, d_partial);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    total_ms += ms;
  }
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);
  float avg_ms = total_ms / (float)timed_iters;

  float sum;
  cudaMemcpy(&sum, d_partial, sizeof(float), cudaMemcpyDeviceToHost);

  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  printf("n = %zu\n", n);
  printf("blocks = %d\n", blocks);
  printf("threads = %d\n", threads);
  printf("SMs = %d\n", num_sms);
  printf("access_pattern = %s\n", contiguous ? "contiguous" : "grid-stride");
  printf("sum = %.10g\n", (double)sum);
  printf("avg_kernel_ms = %.4f\n", avg_ms);
  printf("niterations = %d\n", timed_iters);

  // Verification: reference = sequential double sum cast to float (same as GPU).
  // Small residual difference possible from parallel vs sequential order in double.
  double ref_d = 0.0;
  for (size_t i = 0; i < n; i++) ref_d += (double)h_input[i];
  float ref = (float)ref_d;
  float err = std::fabs(sum - ref);
  float tol = 1e-3f * (std::fabs(ref) + 1.f);  // same scale as CUB/Triton for comparable reporting
  int ok = (err <= tol);
  printf("err = %.6g\n", (double)err);
  printf("tol = %.6g\n", (double)tol);
  if (ok)
    printf("VERIFY PASS\n");
  else
    printf("VERIFY FAIL: gpu=%.10g ref=%.10g\n", (double)sum, (double)ref);

  cudaFree(d_partial);
  cudaFree(d_input);
  free(h_input);
  return ok ? 0 : 1;
}
