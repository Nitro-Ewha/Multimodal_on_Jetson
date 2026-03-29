/***************************************************************************************************
 * benchmark_qkv.cu
 *
 * Compares two approaches:
 *   V1 (Unfused): CUTLASS Batched GEMM → [M, 3HD] → separate CUDA kernel for split+transpose
 *   V2 (Fused):   CUTLASS GEMM + epilogue visitor QKV split+transpose (single kernel)
 *
 * Build (from cutlass/build):
 *   cmake --build . --target 35_benchmark_qkv -j$(nproc)
 *
 * Run:
 *   ./examples/35_gemm_softmax/35_benchmark_qkv \
 *       --seq_len=1024 --hidden=4096 --num_heads=32 --head_dim=128 --batch_count=4
 *
 **************************************************************************************************/

#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// V1: unfused — standard CUTLASS batched GEMM
#include "cutlass/gemm/device/gemm_batched.h"

// V2: fused — our epilogue visitor version
#include "gemm_with_qkv_split.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// V1: Separate split+transpose kernel
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel: [batch, M, 3*H*D] row-major → Q,K,V each [batch, H, M, D]
template <typename Element>
__global__ void split_transpose_kernel(
    Element const* __restrict__ src,     // [batch*M, 3*H*D] row-major
    Element* __restrict__ dst_Q,         // [batch, H, M, D]
    Element* __restrict__ dst_K,
    Element* __restrict__ dst_V,
    int M, int H, int D,
    int64_t src_batch_stride,            // M * 3*H*D
    int64_t dst_batch_stride             // H * M * D
) {
    int hd = H * D;
    int N  = 3 * hd;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    int batch_idx = blockIdx.y;

    if (tid >= total) return;

    int m = tid / N;
    int n = tid % N;

    Element val = src[batch_idx * src_batch_stride + m * N + n];

    int qkv_idx  = n / hd;
    int n_local  = n - qkv_idx * hd;
    int head_idx = n_local / D;
    int dim_idx  = n_local - head_idx * D;

    int out_offset = batch_idx * dst_batch_stride
                   + head_idx * (M * D)
                   + m * D
                   + dim_idx;

    switch (qkv_idx) {
        case 0: dst_Q[out_offset] = val; break;
        case 1: dst_K[out_offset] = val; break;
        default: dst_V[out_offset] = val; break;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {
    bool help;
    int seq_len;
    int hidden_dim;
    int num_heads;
    int head_dim;
    int batch_count;
    int warmup;
    int iterations;
    float alpha;
    float beta;

    Options():
        help(false),
        seq_len(512),
        hidden_dim(2048),
        num_heads(16),
        head_dim(128),
        batch_count(2),
        warmup(10),
        iterations(50),
        alpha(1.0f),
        beta(0.0f)
    { }

    int N() const { return 3 * num_heads * head_dim; }

    void parse(int argc, char const **args) {
        cutlass::CommandLine cmd(argc, args);
        if (cmd.check_cmd_line_flag("help")) { help = true; }
        cmd.get_cmd_line_argument("seq_len", seq_len);
        cmd.get_cmd_line_argument("hidden", hidden_dim);
        cmd.get_cmd_line_argument("num_heads", num_heads);
        cmd.get_cmd_line_argument("head_dim", head_dim);
        cmd.get_cmd_line_argument("batch_count", batch_count);
        cmd.get_cmd_line_argument("warmup", warmup);
        cmd.get_cmd_line_argument("iterations", iterations);
        cmd.get_cmd_line_argument("alpha", alpha);
        cmd.get_cmd_line_argument("beta", beta);
    }

    std::ostream & print_usage(std::ostream &out) const {
        out << "benchmark_qkv — Compare unfused vs fused GEMM+QKV split/transpose\n\n"
            << "Options:\n"
            << "  --seq_len=<int>       Sequence length (M)     [" << seq_len << "]\n"
            << "  --hidden=<int>        Hidden dim (K)          [" << hidden_dim << "]\n"
            << "  --num_heads=<int>     Number of heads         [" << num_heads << "]\n"
            << "  --head_dim=<int>      Head dimension          [" << head_dim << "]\n"
            << "  --batch_count=<int>   Batch size              [" << batch_count << "]\n"
            << "  --warmup=<int>        Warmup iterations       [" << warmup << "]\n"
            << "  --iterations=<int>    Benchmark iterations    [" << iterations << "]\n"
            << "\nExample:\n"
            << "  ./35_benchmark_qkv --seq_len=1024 --hidden=4096 --num_heads=32 --head_dim=128\n\n";
        return out;
    }

    bool supported() const {
        cudaDeviceProp props;
        cudaError_t err = cudaGetDeviceProperties(&props, 0);
        if (err != cudaSuccess) { std::cerr << "No GPU found\n"; return false; }
        if ((props.major * 10 + props.minor) < 80) {
            std::cerr << "Requires SM80+\n"; return false;
        }
        return true;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct BenchmarkResult {
    float gemm_ms;
    float split_ms;
    float total_ms;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Benchmark {

    using ElementA       = cutlass::half_t;
    using ElementB       = cutlass::half_t;
    using ElementC       = cutlass::half_t;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using OperatorClass    = cutlass::arch::OpClassTensorOp;
    using ArchTag          = cutlass::arch::Sm80;
    static int const kStages = 3;

    using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementCompute,
        ElementCompute
    >;

    // ── V1: Standard batched GEMM ──
    using GemmBatched = cutlass::gemm::device::GemmBatched<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementCompute,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueFunctorOp,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        kStages
    >;

    // ── V2: Fused GEMM + QKV split ──
    using GemmFused = cutlass::GemmQKVSplit<
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

    //
    // Data
    //
    Options const &options;
    cutlass::gemm::GemmCoord problem;

    cutlass::DeviceAllocation<ElementA> block_A;
    cutlass::DeviceAllocation<ElementB> block_B;
    cutlass::DeviceAllocation<ElementC> block_C;
    cutlass::DeviceAllocation<ElementC> block_D;    // V1 GEMM output
    cutlass::DeviceAllocation<ElementC> block_Q_v1, block_K_v1, block_V_v1;
    cutlass::DeviceAllocation<ElementC> block_Q_v2, block_K_v2, block_V_v2;

    int64_t lda, ldb, ldc;
    int64_t batch_stride_A, batch_stride_B, batch_stride_C;
    int64_t batch_stride_QKV;

    Benchmark(Options const &opts):
        options(opts),
        problem({opts.seq_len, opts.N(), opts.hidden_dim})
    {
        int M = problem.m(), N = problem.n(), K = problem.k();
        int H = opts.num_heads, D = opts.head_dim;
        int B = opts.batch_count;

        lda = LayoutA::packed({M, K}).stride(0);
        ldb = LayoutB::packed({K, N}).stride(0);
        ldc = LayoutC::packed({M, N}).stride(0);

        batch_stride_A   = (int64_t)M * K;
        batch_stride_B   = (int64_t)K * N;
        batch_stride_C   = (int64_t)M * N;
        batch_stride_QKV = (int64_t)H * M * D;

        block_A.reset(batch_stride_A * B);
        block_B.reset(batch_stride_B * B);
        block_C.reset(batch_stride_C * B);
        block_D.reset(batch_stride_C * B);
        block_Q_v1.reset(batch_stride_QKV * B);
        block_K_v1.reset(batch_stride_QKV * B);
        block_V_v1.reset(batch_stride_QKV * B);
        block_Q_v2.reset(batch_stride_QKV * B);
        block_K_v2.reset(batch_stride_QKV * B);
        block_V_v2.reset(batch_stride_QKV * B);

        // Initialize with random data
        uint64_t seed = 2024;
        cutlass::reference::device::BlockFillRandomUniform(block_A.get(), block_A.size(), seed + 0, ElementA(2), ElementA(-2), 0);
        cutlass::reference::device::BlockFillRandomUniform(block_B.get(), block_B.size(), seed + 1, ElementB(2), ElementB(-2), 0);
        cudaMemset(block_C.get(), 0, block_C.size() * sizeof(ElementC));
    }

    /// V1: GEMM + separate split/transpose
    BenchmarkResult run_unfused(bool do_warmup = true) {

        int M = problem.m(), N = problem.n(), K = problem.k();
        int H = options.num_heads, D = options.head_dim;
        int B = options.batch_count;

        // ── Setup GEMM ──
        GemmBatched gemm_op;

        typename GemmBatched::Arguments gemm_args(
            problem,
            {block_A.get(), lda},  batch_stride_A,
            {block_B.get(), ldb},  batch_stride_B,
            {block_C.get(), ldc},  batch_stride_C,
            {block_D.get(), ldc},  batch_stride_C,
            {options.alpha, options.beta},
            B
        );

        cutlass::Status status = gemm_op.initialize(gemm_args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "V1 GEMM init failed\n";
            return {-1, -1, -1};
        }

        // ── Setup split kernel ──
        int total_elems_per_batch = M * N;
        int threads = 256;
        int blocks = (total_elems_per_batch + threads - 1) / threads;
        dim3 grid(blocks, B);

        // ── Warmup ──
        if (do_warmup) {
            for (int i = 0; i < options.warmup; ++i) {
                gemm_op();
                split_transpose_kernel<<<grid, threads>>>(
                    block_D.get(), block_Q_v1.get(), block_K_v1.get(), block_V_v1.get(),
                    M, H, D, batch_stride_C, batch_stride_QKV);
            }
            cudaDeviceSynchronize();
        }

        // ── Benchmark GEMM only ──
        cudaEvent_t t0, t1, t2;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventCreate(&t2);

        cudaEventRecord(t0);
        for (int i = 0; i < options.iterations; ++i) {
            gemm_op();
        }
        cudaEventRecord(t1);

        for (int i = 0; i < options.iterations; ++i) {
            split_transpose_kernel<<<grid, threads>>>(
                block_D.get(), block_Q_v1.get(), block_K_v1.get(), block_V_v1.get(),
                M, H, D, batch_stride_C, batch_stride_QKV);
        }
        cudaEventRecord(t2);
        cudaEventSynchronize(t2);

        float gemm_ms, split_ms;
        cudaEventElapsedTime(&gemm_ms, t0, t1);
        cudaEventElapsedTime(&split_ms, t1, t2);

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaEventDestroy(t2);

        // ── Benchmark total (GEMM + split together) ──
        cudaEvent_t ta, tb;
        cudaEventCreate(&ta);
        cudaEventCreate(&tb);

        cudaEventRecord(ta);
        for (int i = 0; i < options.iterations; ++i) {
            gemm_op();
            split_transpose_kernel<<<grid, threads>>>(
                block_D.get(), block_Q_v1.get(), block_K_v1.get(), block_V_v1.get(),
                M, H, D, batch_stride_C, batch_stride_QKV);
        }
        cudaEventRecord(tb);
        cudaEventSynchronize(tb);

        float total_ms;
        cudaEventElapsedTime(&total_ms, ta, tb);

        cudaEventDestroy(ta);
        cudaEventDestroy(tb);

        return {
            gemm_ms / options.iterations,
            split_ms / options.iterations,
            total_ms / options.iterations
        };
    }

    /// V2: Fused GEMM + QKV split
    float run_fused(bool do_warmup = true) {

        int M = problem.m(), N = problem.n(), K = problem.k();

        GemmFused fused_op;

        typename GemmFused::Arguments args(
            problem,
            options.batch_count,
            {block_A.get(), lda},
            {block_B.get(), ldb},
            {block_C.get(), ldc},
            {block_D.get(), ldc},  // dummy output
            {options.alpha, options.beta},
            block_Q_v2.get(),
            block_K_v2.get(),
            block_V_v2.get(),
            options.num_heads,
            options.head_dim,
            batch_stride_A,
            batch_stride_B,
            batch_stride_C,
            batch_stride_C,
            batch_stride_QKV
        );

        cutlass::Status status = fused_op.initialize(args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "V2 fused init failed\n";
            return -1;
        }

        // Warmup
        if (do_warmup) {
            for (int i = 0; i < options.warmup; ++i) {
                fused_op();
            }
            cudaDeviceSynchronize();
        }

        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int i = 0; i < options.iterations; ++i) {
            fused_op();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return elapsed_ms / options.iterations;
    }

    /// Verify V1 and V2 produce the same results
    bool verify() {
        // Run both once
        {
            int M = problem.m(), N = problem.n(), H = options.num_heads, D = options.head_dim;
            int B = options.batch_count;

            GemmBatched gemm_op;
            typename GemmBatched::Arguments gemm_args(
                problem,
                {block_A.get(), lda}, batch_stride_A,
                {block_B.get(), ldb}, batch_stride_B,
                {block_C.get(), ldc}, batch_stride_C,
                {block_D.get(), ldc}, batch_stride_C,
                {options.alpha, options.beta},
                B
            );
            gemm_op.initialize(gemm_args);
            gemm_op();

            int total = M * N;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            split_transpose_kernel<<<dim3(blocks, B), threads>>>(
                block_D.get(), block_Q_v1.get(), block_K_v1.get(), block_V_v1.get(),
                M, H, D, batch_stride_C, batch_stride_QKV);

            cudaDeviceSynchronize();
        }

        {
            GemmFused fused_op;
            typename GemmFused::Arguments args(
                problem, options.batch_count,
                {block_A.get(), lda},
                {block_B.get(), ldb},
                {block_C.get(), ldc},
                {block_D.get(), ldc},
                {options.alpha, options.beta},
                block_Q_v2.get(), block_K_v2.get(), block_V_v2.get(),
                options.num_heads, options.head_dim,
                batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_C, batch_stride_QKV
            );
            fused_op.initialize(args);
            fused_op();
            cudaDeviceSynchronize();
        }

        // Compare on host
        int64_t total = batch_stride_QKV * options.batch_count;
        std::vector<ElementC> h_Q1(total), h_K1(total), h_V1(total);
        std::vector<ElementC> h_Q2(total), h_K2(total), h_V2(total);

        cudaMemcpy(h_Q1.data(), block_Q_v1.get(), total * sizeof(ElementC), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_K1.data(), block_K_v1.get(), total * sizeof(ElementC), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_V1.data(), block_V_v1.get(), total * sizeof(ElementC), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Q2.data(), block_Q_v2.get(), total * sizeof(ElementC), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_K2.data(), block_K_v2.get(), total * sizeof(ElementC), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_V2.data(), block_V_v2.get(), total * sizeof(ElementC), cudaMemcpyDeviceToHost);

        float tol = 0.05f;  // half precision tolerance
        int errors = 0;
        for (int64_t i = 0; i < total; ++i) {
            float diff_q = std::abs(float(h_Q1[i]) - float(h_Q2[i]));
            float diff_k = std::abs(float(h_K1[i]) - float(h_K2[i]));
            float diff_v = std::abs(float(h_V1[i]) - float(h_V2[i]));
            if (diff_q > tol || diff_k > tol || diff_v > tol) {
                if (errors < 5) {
                    std::cerr << "  Mismatch at [" << i << "]: "
                              << "Q(" << float(h_Q1[i]) << " vs " << float(h_Q2[i]) << ") "
                              << "K(" << float(h_K1[i]) << " vs " << float(h_K2[i]) << ") "
                              << "V(" << float(h_V1[i]) << " vs " << float(h_V2[i]) << ")\n";
                }
                errors++;
            }
        }

        if (errors > 0) {
            std::cerr << "  Total mismatches: " << errors << " / " << total << "\n";
            return false;
        }
        return true;
    }

    void run() {
        int M = problem.m(), N = problem.n(), K = problem.k();
        int H = options.num_heads, D = options.head_dim;
        int B = options.batch_count;

        double flops = 2.0 * M * N * K * B;
        double bytes_gemm_out = (double)M * N * sizeof(ElementC) * B;
        double bytes_qkv_out  = 3.0 * H * M * D * sizeof(ElementC) * B;

        // ── Header ──
        std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║        GEMM + QKV Split/Transpose Benchmark                  ║\n";
        std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  GEMM: [" << M << " x " << K << "] × [" << K << " x " << N << "]"
                  << std::string(std::max(0, 37 - 20 - (int)std::to_string(M).size()
                    - (int)std::to_string(K).size() - (int)std::to_string(N).size()), ' ')
                  << "║\n";
        std::cout << "║  Heads: " << H << "   HeadDim: " << D
                  << "   Batch: " << B
                  << std::string(std::max(0, 40 - (int)std::to_string(H).size()
                    - (int)std::to_string(D).size() - (int)std::to_string(B).size()), ' ')
                  << "║\n";
        std::cout << "║  N = 3 × " << H << " × " << D << " = " << N
                  << std::string(std::max(0, 44 - (int)std::to_string(H).size()
                    - (int)std::to_string(D).size() - (int)std::to_string(N).size()), ' ')
                  << "║\n";
        std::cout << "║  GFLOPS: " << std::fixed << std::setprecision(1) << flops / 1e9
                  << std::string(std::max(0, 49 - 8), ' ')
                  << "║\n";
        std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";

        // ── Verify ──
        std::cout << "║  Verifying V1 == V2 ...                                      ║\n";
        bool ok = verify();
        std::cout << "║  Result: " << (ok ? "PASSED ✓" : "FAILED ✗")
                  << std::string(ok ? 48 : 48, ' ') << "║\n";
        std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";

        // ── Benchmark V1 ──
        std::cout << "║  Running V1 (GEMM + split kernel) ...                        ║\n";
        auto v1 = run_unfused();

        // ── Benchmark V2 ──
        std::cout << "║  Running V2 (Fused epilogue) ...                             ║\n";
        float v2_ms = run_fused();

        // ── Results ──
        float speedup = v1.total_ms / v2_ms;
        float saved = (v1.total_ms - v2_ms);
        float saved_pct = (1.0f - v2_ms / v1.total_ms) * 100.0f;

        double tflops_v1 = (flops / (v1.total_ms * 1e-3)) / 1e12;
        double tflops_v2 = (flops / (v2_ms * 1e-3)) / 1e12;

        std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║                       Results                                ║\n";
        std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";
        std::cout << std::fixed << std::setprecision(3);

        std::cout << "║                                                              ║\n";
        std::cout << "║  V1 (Unfused):                                               ║\n";
        std::cout << "║    GEMM only:         " << std::setw(8) << v1.gemm_ms
                  << " ms                              ║\n";
        std::cout << "║    Split/Transpose:   " << std::setw(8) << v1.split_ms
                  << " ms                              ║\n";
        std::cout << "║    Total:             " << std::setw(8) << v1.total_ms
                  << " ms    (" << std::setprecision(2) << tflops_v1 << " TFLOPS)          ║\n";
        std::cout << "║                                                              ║\n";
        std::cout << std::setprecision(3);
        std::cout << "║  V2 (Fused):                                                 ║\n";
        std::cout << "║    Single kernel:     " << std::setw(8) << v2_ms
                  << " ms    (" << std::setprecision(2) << tflops_v2 << " TFLOPS)          ║\n";
        std::cout << "║                                                              ║\n";
        std::cout << std::setprecision(2);
        std::cout << "║  Speedup: " << speedup << "x"
                  << "   Saved: " << std::setprecision(3) << saved << " ms"
                  << " (" << std::setprecision(1) << saved_pct << "%%)"
                  << std::string(std::max(0, 20), ' ') << "║\n";
        std::cout << "║                                                              ║\n";
        std::cout << "║  Memory saved: "
                  << std::setprecision(2) << bytes_gemm_out / (1024.0 * 1024.0) << " MB"
                  << " (eliminated D intermediate)             ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
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

    if (!options.supported()) return 0;

    int N = options.N();
    int kElementsPerAccess = 128 / cutlass::sizeof_bits<cutlass::half_t>::value;
    if (N % kElementsPerAccess != 0) {
        std::cerr << "N (= " << N << ") must be divisible by " << kElementsPerAccess << "\n";
        return -1;
    }

    // Print GPU info
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "GPU: " << props.name << "\n\n";

    Benchmark bench(options);
    bench.run();

    return 0;
}

