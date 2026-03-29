/***************************************************************************************************
 * gemm_qkv_split.cu
 *
 * Demonstrates GEMM + QKV Split/Transpose epilogue fusion using CUTLASS 2.x visitor pattern.
 *
 * GEMM:  Input[seq_len, hidden] x Weight[hidden, 3*H*D] -> [seq_len, 3*H*D]
 * Fused: -> Q[H, seq_len, D], K[H, seq_len, D], V[H, seq_len, D]
 *
 * Build (from cutlass/build):
 *   cmake --build . --target 35_gemm_qkv_split -j$(nproc)
 *
 * Run:
 *   ./examples/35_gemm_softmax/35_gemm_qkv_split \
 *       --seq_len=256 --hidden=512 --num_heads=8 --head_dim=64
 *
 **************************************************************************************************/

#include <cmath>
#include <iostream>
#include <vector>
#include <limits>
#include <random>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "gemm_with_qkv_split.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {
  bool help;
  int seq_len;
  int hidden_dim;
  int num_heads;
  int head_dim;
  int batch_count;
  int iterations;
  float alpha;
  float beta;
  bool verification_enabled;
  float tolerance;

  Options():
    help(false),
    seq_len(128),
    hidden_dim(256),
    num_heads(4),
    head_dim(64),
    batch_count(1),
    iterations(20),
    alpha(1.0f),
    beta(0.0f),
    verification_enabled(true),
    tolerance(1e-2f)
  { }

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) { help = true; }

    cmd.get_cmd_line_argument("seq_len", seq_len);
    cmd.get_cmd_line_argument("hidden", hidden_dim);
    cmd.get_cmd_line_argument("num_heads", num_heads);
    cmd.get_cmd_line_argument("head_dim", head_dim);
    cmd.get_cmd_line_argument("batch_count", batch_count);
    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("verify", verification_enabled);
    cmd.get_cmd_line_argument("tolerance", tolerance);
  }

  int N() const { return 3 * num_heads * head_dim; }

  std::ostream & print_usage(std::ostream &out) const {
    out << "gemm_qkv_split example\n\n"
      << "  GEMM + QKV Split/Transpose epilogue fusion.\n\n"
      << "Options:\n\n"
      << "  --help                      Display this message\n"
      << "  --seq_len=<int>             Sequence length (M dimension)     [" << seq_len << "]\n"
      << "  --hidden=<int>              Hidden dimension (K dimension)    [" << hidden_dim << "]\n"
      << "  --num_heads=<int>           Number of attention heads         [" << num_heads << "]\n"
      << "  --head_dim=<int>            Head dimension                    [" << head_dim << "]\n"
      << "  --batch_count=<int>         Batch size                        [" << batch_count << "]\n"
      << "  --alpha=<f32>               Alpha scaling                     [" << alpha << "]\n"
      << "  --beta=<f32>                Beta scaling                      [" << beta << "]\n"
      << "  --iterations=<int>          Profiling iterations              [" << iterations << "]\n"
      << "  --verify=<bool>             Enable verification               [" << verification_enabled << "]\n"
      << "  --tolerance=<f32>           Verification tolerance            [" << tolerance << "]\n"
      << "\n  N dimension = 3 * num_heads * head_dim = " << N() << "\n"
      << "\nExample:\n"
      << "  ./35_gemm_qkv_split --seq_len=256 --hidden=512 --num_heads=8 --head_dim=64\n\n";
    return out;
  }

  bool supported() const {
    if (!(__CUDACC_VER_MAJOR__ >= 11)) {
      std::cerr << "Requires CUDA 11.0+\n";
      return false;
    }
    cudaDeviceProp props;
    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
      std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(error) << "\n";
      return false;
    }
    if ((props.major * 10 + props.minor) < 80) {
      std::cerr << "Requires SM80 (Ampere) or later. Found SM"
                << props.major << props.minor << "\n";
      return false;
    }
    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Testbed {

  //
  // Type definitions (match example 35 style)
  //

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;

  static int const kStages = 3;

  using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementCompute,
    ElementCompute
  >;

  using GemmQKV = cutlass::GemmQKVSplit<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueFunctorOp,
    kStages
  >;

  using LayoutC = cutlass::layout::RowMajor;

  //
  // Data members
  //

  Options const &options;

  cutlass::gemm::GemmCoord problem;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementC> block_D_dummy;  // not actually written
  cutlass::DeviceAllocation<ElementC> block_Q;
  cutlass::DeviceAllocation<ElementC> block_K;
  cutlass::DeviceAllocation<ElementC> block_V;

  int64_t lda, ldb, ldc;
  int64_t total_elements_A_per_batch;
  int64_t total_elements_B_per_batch;
  int64_t total_elements_C_per_batch;
  int64_t total_elements_QKV_per_batch;

  //
  // Methods
  //

  Testbed(Options const &options_):
    options(options_),
    problem({options_.seq_len, options_.N(), options_.hidden_dim})
  {
    lda = LayoutA::packed({problem.m(), problem.k()}).stride(0);
    ldb = LayoutB::packed({problem.k(), problem.n()}).stride(0);
    ldc = LayoutC::packed({problem.m(), problem.n()}).stride(0);

    total_elements_A_per_batch = problem.m() * problem.k();
    total_elements_B_per_batch = problem.k() * problem.n();
    total_elements_C_per_batch = problem.m() * problem.n();
    total_elements_QKV_per_batch = options.num_heads * options.seq_len * options.head_dim;

    int64_t total_A = total_elements_A_per_batch * options.batch_count;
    int64_t total_B = total_elements_B_per_batch * options.batch_count;
    int64_t total_C = total_elements_C_per_batch * options.batch_count;
    int64_t total_QKV = total_elements_QKV_per_batch * options.batch_count;

    block_A.reset(total_A);
    block_B.reset(total_B);
    block_C.reset(total_C);
    block_D_dummy.reset(total_C);
    block_Q.reset(total_QKV);
    block_K.reset(total_QKV);
    block_V.reset(total_QKV);
  }

  /// Initialize random data
  void initialize() {
    uint64_t seed = 2024;

    cutlass::reference::device::BlockFillRandomUniform(
      block_A.get(), block_A.size(), seed + 0, ElementA(2), ElementA(-2), 0);

    cutlass::reference::device::BlockFillRandomUniform(
      block_B.get(), block_B.size(), seed + 1, ElementB(2), ElementB(-2), 0);

    cutlass::reference::device::BlockFillRandomUniform(
      block_C.get(), block_C.size(), seed + 2, ElementC(2), ElementC(-2), 0);

    cudaMemset(block_Q.get(), 0, block_Q.size() * sizeof(ElementC));
    cudaMemset(block_K.get(), 0, block_K.size() * sizeof(ElementC));
    cudaMemset(block_V.get(), 0, block_V.size() * sizeof(ElementC));
    cudaMemset(block_D_dummy.get(), 0, block_D_dummy.size() * sizeof(ElementC));
  }

  /// Verify results against CPU reference
  bool verify() {
    int M = options.seq_len;
    int N = options.N();
    int K = options.hidden_dim;
    int H = options.num_heads;
    int D = options.head_dim;

    // Copy device data to host
    std::vector<ElementA> h_A(block_A.size());
    std::vector<ElementB> h_B(block_B.size());
    std::vector<ElementC> h_C(block_C.size());
    std::vector<ElementC> h_Q(block_Q.size());
    std::vector<ElementC> h_K(block_K.size());
    std::vector<ElementC> h_V(block_V.size());

    cudaMemcpy(h_A.data(), block_A.get(), block_A.size() * sizeof(ElementA), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B.data(), block_B.get(), block_B.size() * sizeof(ElementB), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C.data(), block_C.get(), block_C.size() * sizeof(ElementC), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Q.data(), block_Q.get(), block_Q.size() * sizeof(ElementC), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_K.data(), block_K.get(), block_K.size() * sizeof(ElementC), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V.data(), block_V.get(), block_V.size() * sizeof(ElementC), cudaMemcpyDeviceToHost);

    bool passed = true;

    for (int b = 0; b < options.batch_count; ++b) {

      // 1. CPU reference GEMM: D_ref = alpha * A * B + beta * C
      std::vector<float> D_ref(M * N, 0.0f);

      ElementA const *ptr_A = h_A.data() + b * total_elements_A_per_batch;
      ElementB const *ptr_B = h_B.data() + b * total_elements_B_per_batch;
      ElementC const *ptr_C = h_C.data() + b * total_elements_C_per_batch;

      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          float accum = 0.0f;
          for (int k = 0; k < K; ++k) {
            // A is RowMajor [M, K]
            float a_val = float(ptr_A[m * K + k]);
            // B is ColumnMajor [K, N] → element (k, n) at index n * K + k
            float b_val = float(ptr_B[n * K + k]);
            accum += a_val * b_val;
          }
          D_ref[m * N + n] = options.alpha * accum + options.beta * float(ptr_C[m * N + n]);
        }
      }

      // 2. CPU reference split + transpose
      //    D_ref[M, 3*H*D] -> Q_ref[H, M, D], K_ref[H, M, D], V_ref[H, M, D]
      int hd = H * D;
      std::vector<float> Q_ref(H * M * D);
      std::vector<float> K_ref(H * M * D);
      std::vector<float> V_ref(H * M * D);

      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          float val = D_ref[m * N + n];
          int qkv_idx = n / hd;
          int n_local = n % hd;
          int head_idx = n_local / D;
          int dim_idx  = n_local % D;

          int out_offset = head_idx * (M * D) + m * D + dim_idx;

          switch (qkv_idx) {
            case 0: Q_ref[out_offset] = val; break;
            case 1: K_ref[out_offset] = val; break;
            case 2: V_ref[out_offset] = val; break;
          }
        }
      }

      // 3. Compare
      ElementC const *gpu_Q = h_Q.data() + b * total_elements_QKV_per_batch;
      ElementC const *gpu_K = h_K.data() + b * total_elements_QKV_per_batch;
      ElementC const *gpu_V = h_V.data() + b * total_elements_QKV_per_batch;

      int errors_q = 0, errors_k = 0, errors_v = 0;

      for (int i = 0; i < H * M * D; ++i) {
        if (std::abs(float(gpu_Q[i]) - Q_ref[i]) > options.tolerance) {
          if (errors_q < 5) {
            std::cerr << "  Q mismatch at [" << i << "]: GPU=" << float(gpu_Q[i])
                      << " ref=" << Q_ref[i] << "\n";
          }
          errors_q++;
        }
        if (std::abs(float(gpu_K[i]) - K_ref[i]) > options.tolerance) {
          if (errors_k < 5) {
            std::cerr << "  K mismatch at [" << i << "]: GPU=" << float(gpu_K[i])
                      << " ref=" << K_ref[i] << "\n";
          }
          errors_k++;
        }
        if (std::abs(float(gpu_V[i]) - V_ref[i]) > options.tolerance) {
          if (errors_v < 5) {
            std::cerr << "  V mismatch at [" << i << "]: GPU=" << float(gpu_V[i])
                      << " ref=" << V_ref[i] << "\n";
          }
          errors_v++;
        }
      }

      int total_elems = H * M * D;
      if (errors_q > 0 || errors_k > 0 || errors_v > 0) {
        std::cerr << "Batch " << b << " FAILED: Q errors=" << errors_q
                  << "/" << total_elems
                  << "  K errors=" << errors_k << "/" << total_elems
                  << "  V errors=" << errors_v << "/" << total_elems << "\n";
        passed = false;
      } else {
        std::cout << "Batch " << b << ": PASSED (all " << total_elems << " elements match)\n";
      }
    }

    return passed;
  }

  /// Run the fused kernel
  bool run() {
    initialize();

    std::cout << "Problem: GEMM [" << problem.m() << ", " << problem.n()
              << "] = [" << problem.m() << ", " << problem.k() << "] x ["
              << problem.k() << ", " << problem.n() << "]\n";
    std::cout << "QKV split: num_heads=" << options.num_heads
              << "  head_dim=" << options.head_dim
              << "  batch=" << options.batch_count << "\n";
    std::cout << "Output: Q,K,V each [" << options.num_heads << ", "
              << options.seq_len << ", " << options.head_dim << "]\n\n";

    // Create the GemmQKVSplit operator
    GemmQKV gemm_qkv;

    typename GemmQKV::Arguments args(
      problem,
      options.batch_count,
      // A, B, C, D
      {block_A.get(), lda},
      {block_B.get(), ldb},
      {block_C.get(), ldc},
      {block_D_dummy.get(), ldc},
      // linear scaling
      {options.alpha, options.beta},
      // QKV output pointers
      block_Q.get(),
      block_K.get(),
      block_V.get(),
      options.num_heads,
      options.head_dim,
      // batch strides
      total_elements_A_per_batch,
      total_elements_B_per_batch,
      total_elements_C_per_batch,
      total_elements_C_per_batch,
      total_elements_QKV_per_batch
    );

    cutlass::Status status = gemm_qkv.initialize(args);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize GemmQKVSplit\n";
      return false;
    }

    // Run once to verify
    status = gemm_qkv.run();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run GemmQKVSplit\n";
      return false;
    }
    cudaDeviceSynchronize();

    // Verify
    bool passed = true;
    if (options.verification_enabled) {
      std::cout << "Verifying...\n";
      passed = verify();
      std::cout << "\n";
    }

    // Profile
    if (options.iterations > 0) {
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start);
      for (int i = 0; i < options.iterations; ++i) {
        gemm_qkv.run();
      }
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);

      float elapsed_ms;
      cudaEventElapsedTime(&elapsed_ms, start, stop);
      float avg_ms = elapsed_ms / options.iterations;

      // Compute throughput
      double flops = 2.0 * problem.m() * problem.n() * problem.k() * options.batch_count;
      double tflops = (flops / (avg_ms * 1e-3)) / 1e12;

      double bytes_read  = (double)(problem.m() * problem.k() + problem.k() * problem.n())
                         * sizeof(ElementA) * options.batch_count;
      double bytes_write = (double)(3 * options.num_heads * options.seq_len * options.head_dim)
                         * sizeof(ElementC) * options.batch_count;
      double bw_gbps = (bytes_read + bytes_write) / (avg_ms * 1e-3) / 1e9;

      std::cout << "Performance:\n"
                << "  Avg time:    " << avg_ms << " ms\n"
                << "  TFLOPS:      " << tflops << "\n"
                << "  Bandwidth:   " << bw_gbps << " GB/s\n";

      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  Options options;
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout);
    return 0;
  }

  if (!options.supported()) {
    return 0;
  }

  // Validate dimensions
  int N = options.N();
  int kElementsPerAccess = 128 / cutlass::sizeof_bits<cutlass::half_t>::value; // = 8
  if (N % kElementsPerAccess != 0) {
    std::cerr << "N (= 3 * num_heads * head_dim = " << N
              << ") must be divisible by " << kElementsPerAccess << "\n";
    return -1;
  }

  std::cout << "===========================================\n";
  std::cout << " GEMM + QKV Split/Transpose Fusion Example\n";
  std::cout << "===========================================\n\n";

  Testbed testbed(options);
  bool passed = testbed.run();

  std::cout << "\n" << (passed ? "PASSED" : "FAILED") << "\n";

  return passed ? 0 : -1;
}
