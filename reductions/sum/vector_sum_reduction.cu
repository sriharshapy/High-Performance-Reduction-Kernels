/*
 * Fast 1D vector sum reduction (CUDA).
 *
 * For n > 100 million uses CUB DeviceReduce::Sum; otherwise custom two-pass
 * reduction: strided per-thread accumulation → warp shuffle reduction
 * → shared memory for block partial sums → second pass to single scalar.
 * Based on:
 *   - Ash Vardanian, ParallelReductionsBenchmark (reduce_cuda.cuh)
 *     https://github.com/ashvardanian/ParallelReductionsBenchmark
 *   - NVIDIA "Faster Parallel Reductions on Kepler" / warp-level primitives
 *     https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
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
#include <cub/cub.cuh>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu

// Use CUB DeviceReduce::Sum when n exceeds this (100 million).
#define N_CUB_THRESHOLD 100000000UL

// At this element count (~800 MB) the kernel becomes fully DRAM-bandwidth-bound.
// Above the threshold, the single-accumulator dependency chain limits ILP and
// prevents the hardware from issuing enough independent DRAM requests to saturate
// all memory controllers. The ILP4 kernel below breaks that chain.
#define N_ILP4_THRESHOLD 200000000UL

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
// Kernel: accumulation over contiguous segments per block, then block reduce.
// Uses float4 loads for better memory throughput; scalar tail for remainder.
// Each block is assigned a contiguous range of float4-sized chunks, improving
// L1/L2 locality compared to a single global strided pattern.
// Everything is accumulated in float.
// -----------------------------------------------------------------------------
__global__ void reduce_sum_kernel_float_in(const float* __restrict__ inputs,
                                           size_t n,
                                           float* __restrict__ partial_sums) {
  const int  threads_per_block = blockDim.x;
  const int  num_blocks        = gridDim.x;
  const int  tid               = threadIdx.x;
  const int  bid               = blockIdx.x;

  // Number of float4-sized chunks in the input
  const size_t vec_n4 = n / 4;  // each chunk is 4 floats

  // Assign a contiguous range of float4 chunks to each block
  const size_t float4_per_block = (vec_n4 + (size_t)num_blocks - 1) / (size_t)num_blocks;
  const size_t start4 = (size_t)bid * float4_per_block;
  size_t       end4   = start4 + float4_per_block;
  if (end4 > vec_n4) end4 = vec_n4;

  float sum = 0.0f;

  const float4* inputs4 = reinterpret_cast<const float4*>(inputs);
  // Vectorized path: this block walks its own contiguous segment,
  // threads iterate over that segment with a block-local stride.
  for (size_t idx4 = start4 + (size_t)tid; idx4 < end4; idx4 += (size_t)threads_per_block) {
    const float4 v = inputs4[idx4];
    sum += v.x + v.y + v.z + v.w;
  }

  // Scalar tail for remaining elements (n % 4).
  const size_t tail_start    = vec_n4 * 4;
  const size_t total_threads = (size_t)threads_per_block * (size_t)num_blocks;
  const size_t global_tid    = (size_t)bid * (size_t)threads_per_block + (size_t)tid;
  for (size_t i = tail_start + global_tid; i < n; i += total_threads)
    sum += inputs[i];

  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) partial_sums[blockIdx.x] = sum;
}

// -----------------------------------------------------------------------------
// ILP4 kernel: same contiguous-block assignment as the kernel above, but uses
// four independent accumulators fed by four independent __ldg float4 loads per
// loop iteration. The compiler can issue all four loads before any add resolves,
// keeping the DRAM controllers saturated on bandwidth-bound problems (n ≥ 200M).
//
// Used only when n >= N_ILP4_THRESHOLD; below that threshold the shorter startup
// path of reduce_sum_kernel_float_in wins because the array fits in L2/L3 and
// ILP provides no additional benefit.
// -----------------------------------------------------------------------------
__global__ void reduce_sum_kernel_float_in_ilp4(const float* __restrict__ inputs,
                                                size_t n,
                                                float* __restrict__ partial_sums) {
  const int    threads_per_block = blockDim.x;
  const int    num_blocks        = gridDim.x;
  const int    tid               = threadIdx.x;
  const int    bid               = blockIdx.x;
  const size_t stride            = (size_t)threads_per_block;

  const size_t vec_n4           = n / 4;
  const size_t float4_per_block = (vec_n4 + (size_t)num_blocks - 1) / (size_t)num_blocks;
  const size_t start4           = (size_t)bid * float4_per_block;
  size_t       end4             = start4 + float4_per_block;
  if (end4 > vec_n4) end4 = vec_n4;

  const float4* inputs4 = reinterpret_cast<const float4*>(inputs);

  float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
  size_t idx4 = start4 + (size_t)tid;

  // Four independent __ldg loads per iteration; the compiler issues all four
  // before any add resolves, breaking the serial dependency chain and allowing
  // the hardware prefetcher to keep DRAM fully pipelined.
  for (; idx4 + 3 * stride < end4; idx4 += 4 * stride) {
    float4 v0 = __ldg(&inputs4[idx4]);
    float4 v1 = __ldg(&inputs4[idx4 + stride]);
    float4 v2 = __ldg(&inputs4[idx4 + 2 * stride]);
    float4 v3 = __ldg(&inputs4[idx4 + 3 * stride]);
    sum0 += v0.x + v0.y + v0.z + v0.w;
    sum1 += v1.x + v1.y + v1.z + v1.w;
    sum2 += v2.x + v2.y + v2.z + v2.w;
    sum3 += v3.x + v3.y + v3.z + v3.w;
  }
  // Remainder: fewer than 4 strides left in this block's segment.
  for (; idx4 < end4; idx4 += stride) {
    float4 v = __ldg(&inputs4[idx4]);
    sum0 += v.x + v.y + v.z + v.w;
  }

  float sum = sum0 + sum1 + sum2 + sum3;

  // Scalar tail for elements that don't fill a float4 (n % 4).
  const size_t tail_start    = vec_n4 * 4;
  const size_t total_threads = (size_t)threads_per_block * (size_t)num_blocks;
  const size_t global_tid    = (size_t)bid * (size_t)threads_per_block + (size_t)tid;
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
// Host: run two-pass reduction in float, return result.
// When n >= N_CUB_THRESHOLD use CUB; else second pass uses one block with one
// thread per partial (coalesced) to cut DRAM.
// -----------------------------------------------------------------------------
static float reduce_sum(const float* d_inputs, size_t n, int blocks, int threads_per_block,
                        float* d_partial, void* d_cub_temp, size_t cub_temp_bytes) {
  if (n > N_CUB_THRESHOLD && d_cub_temp != nullptr) {
    (void)cub::DeviceReduce::Sum(d_cub_temp, cub_temp_bytes, d_inputs, d_partial, n);
    cudaDeviceSynchronize();
    float result;
    cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
    return result;
  }

  if (n > N_ILP4_THRESHOLD)
    reduce_sum_kernel_float_in_ilp4<<<blocks, threads_per_block>>>(d_inputs, n, d_partial);
  else
    reduce_sum_kernel_float_in<<<blocks, threads_per_block>>>(d_inputs, n, d_partial);

  // One thread per partial, round up to warp size; cap at 1024
  int pass2_threads = (int)((blocks + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
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

  float* d_input   = nullptr;
  float* d_partial = nullptr;
  cudaMalloc(&d_input, n * sizeof(float));
  cudaMalloc(&d_partial, (size_t)max_blocks * sizeof(float));

  void* d_cub_temp = nullptr;
  size_t cub_temp_bytes = 0;
  if (n >= N_CUB_THRESHOLD) {
    cudaError_t err = cub::DeviceReduce::Sum(nullptr, cub_temp_bytes, d_input, d_partial, n);
    if (err != cudaSuccess) { fprintf(stderr, "CUB temp size query failed: %s\n", cudaGetErrorString(err)); cudaFree(d_partial); cudaFree(d_input); free(h_input); return 1; }
    cudaMalloc(&d_cub_temp, cub_temp_bytes);
  }

  cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

  if (do_warmup) {
    const int warmup_iters = 3;
    for (int w = 0; w < warmup_iters; w++)
      (void)reduce_sum(d_input, n, blocks, threads, d_partial, d_cub_temp, cub_temp_bytes);
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
    if (n >= N_CUB_THRESHOLD)
      (void)cub::DeviceReduce::Sum(d_cub_temp, cub_temp_bytes, d_input, d_partial, n);
    else {
      if (n >= N_ILP4_THRESHOLD)
        reduce_sum_kernel_float_in_ilp4<<<blocks, threads>>>(d_input, n, d_partial);
      else
        reduce_sum_kernel_float_in<<<blocks, threads>>>(d_input, n, d_partial);
      reduce_sum_kernel_float_in_pass2<<<1, pass2_threads>>>(d_partial, (size_t)blocks, d_partial);
    }
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
  printf("reduction = %s\n", n >= N_CUB_THRESHOLD ? "CUB" : "custom");
  printf("blocks = %d\n", blocks);
  printf("threads = %d\n", threads);
  printf("SMs = %d\n", num_sms);
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

  if (d_cub_temp) cudaFree(d_cub_temp);
  cudaFree(d_partial);
  cudaFree(d_input);
  free(h_input);
  return ok ? 0 : 1;
}
