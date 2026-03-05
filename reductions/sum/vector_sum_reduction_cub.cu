/*
 * Fast 1D vector sum reduction (CUDA, CUB).
 *
 * Same overall behavior as `reductions/sum/vector_sum_reduction.cu`,
 * but the actual reduction is implemented with CUB primitives instead
 * of hand-written warp- and block-level reductions.
 *
 * Compile (example):
 *   nvcc -O3 -arch=sm_80 -I${CUB_PATH} -o vector_sum_reduction_cub vector_sum_reduction_cub.cu
 *
 * Run:
 *   ./vector_sum_reduction_cub [input_file] [n]
 *   - input_file: raw binary float32 (default: "out.bin"); use first n elements
 *   - n         : number of elements to reduce, i.e. first n from file (default: 1000000000)
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

// -----------------------------------------------------------------------------
// Host helper: run CUB device-wide sum reduction on float32 input.
// Accumulation is in float (CUB's native behavior); verification on host
// uses a double-precision reference with a modest tolerance, matching the
// Triton script's approach.
// -----------------------------------------------------------------------------
static float reduce_sum_cub(const float* d_input,
                            size_t n,
                            void* d_temp_storage,
                            size_t temp_storage_bytes,
                            float* d_output) {
  // Single call; caller is responsible for allocating and reusing temp storage.
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, n);
  return 0.0f;  // dummy; result is written to d_output
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
  if (a < argc && std::strcmp(argv[a], "--no-warmup") == 0) {
    do_warmup = 0;
    a++;
  }
  if (a < argc) path = argv[a++];
  if (a < argc) {
    long long n_arg = std::atoll(argv[a++]);
    if (n_arg <= 0) {
      std::fprintf(stderr, "Error: n must be positive (got %lld)\n", static_cast<long long>(n_arg));
      return 1;
    }
    n = static_cast<size_t>(n_arg);
  }
  if (a < argc) {
    int ti = std::atoi(argv[a++]);
    if (ti <= 0) {
      std::fprintf(stderr, "Error: timed_iters must be positive (got %d)\n", ti);
      return 1;
    }
    timed_iters = ti;
  }

  float* h_input = static_cast<float*>(std::malloc(n * sizeof(float)));
  if (!h_input) {
    std::fprintf(stderr, "malloc failed\n");
    return 1;
  }

  {
    FILE* f = std::fopen(path, "rb");
    if (f) {
      size_t read = std::fread(h_input, sizeof(float), n, f);
      std::fclose(f);
      if (read != n) {
        std::fprintf(stderr, "Error: file has %zu floats, need %zu (first n elements)\n", read, n);
        std::free(h_input);
        return 1;
      }
    } else {
      for (size_t i = 0; i < n; i++)
        h_input[i] = 1.f / static_cast<float>(i + 1);
    }
  }

  // Device buffers
  float* d_input  = nullptr;
  float* d_output = nullptr;
  cudaMalloc(&d_input, n * sizeof(float));
  cudaMalloc(&d_output, sizeof(float));
  cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, n);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  if (do_warmup) {
    const int warmup_iters = 3;
    for (int w = 0; w < warmup_iters; ++w) {
      reduce_sum_cub(d_input, n, d_temp_storage, temp_storage_bytes, d_output);
    }
    cudaDeviceSynchronize();
  }

  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);
  float total_ms = 0.f;
  for (int t = 0; t < timed_iters; ++t) {
    cudaEventRecord(ev_start);
    reduce_sum_cub(d_input, n, d_temp_storage, temp_storage_bytes, d_output);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    total_ms += ms;
  }
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);
  float avg_ms = total_ms / static_cast<float>(timed_iters);

  // Fetch result back to host.
  float sum = 0.f;
  cudaMemcpy(&sum, d_output, sizeof(float), cudaMemcpyDeviceToHost);

  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  std::printf("n = %zu\n", n);
  std::printf("SMs = %d\n", num_sms);
  std::printf("sum = %.10g\n", static_cast<double>(sum));
  std::printf("avg_kernel_ms = %.4f\n", avg_ms);
  std::printf("niterations = %d\n", timed_iters);

  // Verification: double-precision sequential reference on CPU, cast to float.
  double ref_d = 0.0;
  for (size_t i = 0; i < n; ++i)
    ref_d += static_cast<double>(h_input[i]);
  float ref = static_cast<float>(ref_d);
  float err = std::fabs(sum - ref);
  float tol = 1e-3f * (std::fabs(ref) + 1.f);  // same scale as CUDA/Triton for comparable reporting
  int ok = (err <= tol);
  std::printf("err = %.6g\n", static_cast<double>(err));
  std::printf("tol = %.6g\n", static_cast<double>(tol));
  if (ok) {
    std::printf("VERIFY PASS\n");
  } else {
    std::printf("VERIFY FAIL: cub=%.10g ref=%.10g\n",
                static_cast<double>(sum), static_cast<double>(ref));
  }

  cudaFree(d_temp_storage);
  cudaFree(d_output);
  cudaFree(d_input);
  std::free(h_input);
  return ok ? 0 : 1;
}

