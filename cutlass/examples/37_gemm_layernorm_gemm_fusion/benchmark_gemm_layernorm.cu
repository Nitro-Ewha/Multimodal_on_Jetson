/***************************************************************************************************
 * benchmark_gemm_layernorm.cu
 *
 * 두 가지 접근 방식을 비교합니다:
 * V1 (Unfused): CUTLASS GEMM -> [M, N] -> 별도의 Bias+Residual+LayerNorm 커널 호출
 * V2 (Fused):   CUTLASS GEMM + Epilogue Visitor 기반의 단일 파이프라인 (우리의 구현)
 *
 * 빌드 (cutlass/build 디렉토리 기준):
 * cmake --build . --target benchmark_gemm_layernorm -j$(nproc)
 *
 * 실행:
 * ./benchmark_gemm_layernorm --seq_len=1024 --hidden=768 --warmup=10 --iterations=100
 **************************************************************************************************/

#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"

// V1: standard CUTLASS GEMM
#include "cutlass/gemm/device/gemm.h"

// V2: fused — 우리의 융합 버전
#include "gemm_with_layernorm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
// V1용 별도 커널: Bias Add + Residual Add + LayerNorm (Unfused 구현)
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ void bias_residual_layernorm_unfused_kernel(
    T* D, const T* C, const T* Bias, const T* Residual,
    const T* Gamma, const T* Beta, int M, int N) {
    
    int row = blockIdx.x;
    if (row >= M) return;

    // 공유 메모리 선언
    __shared__ float s_sum;
    __shared__ float s_sq_sum;
    
    if (threadIdx.x == 0) {
        s_sum = 0.0f;
        s_sq_sum = 0.0f;
    }
    __syncthreads();

    // 1. 스레드별로 데이터 분산 처리 (병렬화)
    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float val = (float)C[row * N + col] + (float)Bias[col] + (float)Residual[row * N + col];
        thread_sum += val;
        thread_sq_sum += val * val;
    }

    // 부분합을 공유 메모리에 원자적 덧셈
    atomicAdd(&s_sum, thread_sum);
    atomicAdd(&s_sq_sum, thread_sq_sum);
    __syncthreads(); // 모든 스레드의 덧셈이 끝날 때까지 대기

    // 2. 평균과 역표준편차 계산
    float mean = s_sum / N;
    float var = (s_sq_sum / N) - (mean * mean);
    // 🌟 FIX 3: var가 부동소수점 오차로 음수가 되는 것을 방지
    float inv_std = 1.0f / sqrtf(fmaxf(var, 0.0f) + 1e-6f);

    // 3. 최종 연산 및 저장
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float val = (float)C[row * N + col] + (float)Bias[col] + (float)Residual[row * N + col];
        D[row * N + col] = (T)((val - mean) * inv_std * (float)Gamma[col] + (float)Beta[col]);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {
    bool help;
    int batch_size;
    int seq_len;    // M
    int hidden_dim; // N, K
    int warmup;
    int iterations;
    float alpha;
    float beta;

    Options():
        help(false),
        batch_size(1),
        seq_len(1024),
        hidden_dim(768),
        warmup(10),
        iterations(50),
        alpha(1.0f),
        beta(0.0f)
    { }

    void parse(int argc, char const **args) {
        cutlass::CommandLine cmd(argc, args);
        if (cmd.check_cmd_line_flag("help")) { help = true; }
        cmd.get_cmd_line_argument("batch_size", batch_size);
        cmd.get_cmd_line_argument("seq_len", seq_len);
        cmd.get_cmd_line_argument("hidden", hidden_dim);
        cmd.get_cmd_line_argument("warmup", warmup);
        cmd.get_cmd_line_argument("iterations", iterations);
    }
};

struct BenchmarkResult {
    float gemm_ms;
    float post_ms; // bias+res+ln
    float total_ms;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Benchmark {
    using Element = cutlass::half_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    Options const &options;
    
    // D_v1: Unfused 결과 저장, D_v2: Fused 결과 저장
    // D_intermediate: Unfused에서 GEMM 결과 저장 (D_v1과 달리 LayerNorm 적용 전), E_intermediate: Fused에서 Epilogue 직전의 중간 결과 저장
    cutlass::DeviceAllocation<Element> block_A, block_B, block_C, block_Bias, block_Residual, block_D_v1, block_D_v2;
    cutlass::DeviceAllocation<Element> block_D_intermediate;
    cutlass::DeviceAllocation<Element> block_E_intermediate;
    cutlass::DeviceAllocation<Element> block_D_v1_final;
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor_variance;
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor_mean;
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor_gamma;
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor_beta;

    Benchmark(Options const &opts) : options(opts) {
        int M = opts.batch_size * opts.seq_len, N = opts.hidden_dim, K = opts.hidden_dim;

        block_A.reset(M * K);
        block_B.reset(K * N);
        block_C.reset(M * N);
        block_Bias.reset(N);
        block_Residual.reset(M * N);
        block_D_intermediate.reset(M*N);
        block_E_intermediate.reset(K*N);
        block_D_v1.reset(M * N);
        block_D_v2.reset(M * N);
        block_D_v1_final.reset(M * N);
        // tensor_variance.reset({M,1});
        // tensor_mean.reset({M,1});

        // 768(N) / 128(Threadblock_N) = 6개의 부분합 공간 할당
        int partial_sums_count = (N + 127) / 128; 
        tensor_variance.reset({M, partial_sums_count});
        tensor_mean.reset({M, partial_sums_count});

        tensor_gamma.reset({1,N});
        tensor_beta.reset({1,N});

        uint64_t seed = 2024;
        cutlass::reference::device::BlockFillRandomUniform(block_A.get(), M*K, seed, Element(1), Element(-1), 0);
        cutlass::reference::device::BlockFillRandomUniform(block_B.get(), K*N, seed+1, Element(1), Element(-1), 0);
        cutlass::reference::device::BlockFillRandomUniform(block_Bias.get(), N, seed+2, Element(0.5f), Element(-0.5f), 0);
        cutlass::reference::device::BlockFillRandomUniform(block_Residual.get(), M*N, seed+3, Element(0.1f), Element(-0.1f), 0);
        cutlass::reference::device::BlockFillRandomUniform(
            tensor_gamma.device_data(), N, seed+4, Element(1), Element(1), 0);

        cutlass::reference::device::BlockFillRandomUniform(
            tensor_beta.device_data(), N, seed+5, Element(0.01f), Element(-0.01f), 0);
        cutlass::reference::device::BlockFillRandomUniform(
            block_E_intermediate.get(), K*N, seed+6, Element(1), Element(-1), 0);

        // tensor_gamma.sync_device();
        // tensor_beta.sync_device();
        // tensor_mean.sync_device();
        // tensor_variance.sync_device();
    }

    // 🚨 에러 확인용 매크로 추가
    #define CHECK_CUDA_ERROR(step_name) \
        do { \
            cudaError_t err = cudaGetLastError(); \
            if (err != cudaSuccess) { \
                std::cerr << "\n🚨 CUDA Error at [" << step_name << "]: " << cudaGetErrorString(err) << "\n"; \
            } \
        } while(0)

    // --- V1: Unfused Benchmark ---
    BenchmarkResult run_unfused() {
        int M = options.batch_size * options.seq_len, N = options.hidden_dim, K = options.hidden_dim;
        using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<Element, 1, Element, Element>;
        
        // 1차 GEMM 타입 (기존과 동일)
        using GemmUnfused1 = cutlass::gemm::device::Gemm<Element, LayoutA, Element, LayoutB, Element, LayoutC>;
        
        // 🌟 [핵심] 2차 GEMM 타입 추가: 가중치 E를 RowMajor(LayoutA)로 읽어들이도록 설정!
        using GemmUnfused2 = cutlass::gemm::device::Gemm<Element, LayoutA, Element, LayoutA, Element, LayoutC>;

        GemmUnfused1 gemm_op1;
        GemmUnfused2 gemm_op2;

        typename GemmUnfused1::Arguments gemm_args1(
            {M, N, K},
            cutlass::TensorRef<Element, LayoutA>(block_A.get(), LayoutA(K)),
            cutlass::TensorRef<Element, LayoutB>(block_B.get(), LayoutB(K)),
            cutlass::TensorRef<Element, LayoutC>(block_C.get(), LayoutC(N)),
            cutlass::TensorRef<Element, LayoutC>(block_D_intermediate.get(), LayoutC(N)),
            typename EpilogueOutputOp::Params(Element(options.alpha), Element(options.beta))
        );

        typename GemmUnfused2::Arguments gemm_args2(
            {M, N, K},
            cutlass::TensorRef<Element, LayoutA>(block_D_v1.get(), LayoutA(K)),
            cutlass::TensorRef<Element, LayoutA>(block_E_intermediate.get(), LayoutA(N)), // 🌟 LayoutA로 변경, ldm=N
            cutlass::TensorRef<Element, LayoutC>(block_D_v1_final.get(), LayoutC(N)),
            cutlass::TensorRef<Element, LayoutC>(block_D_v1_final.get(), LayoutC(N)),
            typename EpilogueOutputOp::Params(Element(options.alpha), Element(options.beta))
        );

        gemm_op1.initialize(gemm_args1);
        gemm_op2.initialize(gemm_args2);

        // cudaEvent_t t0, t1, t2, t3;
        // cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventCreate(&t2); cudaEventCreate(&t3);

        // Warmup
        gemm_op1();
        cudaDeviceSynchronize();
        bias_residual_layernorm_unfused_kernel<<<M, 256>>>(
            block_D_v1.get(), block_D_intermediate.get(), block_Bias.get(), block_Residual.get(),
            tensor_gamma.device_data(), tensor_beta.device_data(), M, N
        );
        cudaDeviceSynchronize();
        gemm_op2();
        cudaDeviceSynchronize();

        // Run
        int iters = options.iterations;

        // 1. 이벤트 배열(Vector) 생성
        std::vector<cudaEvent_t> t0(iters), t1(iters), t2(iters), t3(iters);
        for (int i = 0; i < iters; ++i) {
            cudaEventCreate(&t0[i]); cudaEventCreate(&t1[i]);
            cudaEventCreate(&t2[i]); cudaEventCreate(&t3[i]);
        }

        // 2. GPU 파이프라인을 깨지 않고 연속으로 기록 (비동기 실행)
        for (int i = 0; i < iters; ++i) {
            cudaEventRecord(t0[i]);
            gemm_op1();
            
            cudaEventRecord(t1[i]);
            bias_residual_layernorm_unfused_kernel<<<M, 256>>>(
                block_D_v1.get(), block_D_intermediate.get(), block_Bias.get(), block_Residual.get(),
                tensor_gamma.device_data(), tensor_beta.device_data(), M, N
            );
            
            cudaEventRecord(t2[i]);
            gemm_op2();
            
            cudaEventRecord(t3[i]);
        }

        // 3. 모든 작업이 끝날 때까지 한 번만 대기
        cudaDeviceSynchronize();

        // 4. 시간 누적 계산
        float total_gemm1_ms = 0.0f, total_ln_ms = 0.0f, total_gemm2_ms = 0.0f;
        for (int i = 0; i < iters; ++i) {
            float ms;
            cudaEventElapsedTime(&ms, t0[i], t1[i]); total_gemm1_ms += ms;
            cudaEventElapsedTime(&ms, t1[i], t2[i]); total_ln_ms += ms;
            cudaEventElapsedTime(&ms, t2[i], t3[i]); total_gemm2_ms += ms;
        }

        // 5. 생성한 이벤트 메모리 해제
        for (int i = 0; i < iters; ++i) {
            cudaEventDestroy(t0[i]); cudaEventDestroy(t1[i]);
            cudaEventDestroy(t2[i]); cudaEventDestroy(t3[i]);
        }

        // 6. 평균 시간 계산 및 반환
        float avg_gemm_ms = (total_gemm1_ms + total_gemm2_ms) / iters;
        float avg_ln_ms = total_ln_ms / iters;
        float avg_total_ms = avg_gemm_ms + avg_ln_ms;

        return { 
            avg_gemm_ms, 
            avg_ln_ms, 
            avg_total_ms 
        };
    
    }

    // --- V2: Fused Benchmark ---
    float run_fused() {
        int M = options.batch_size * options.seq_len, N = options.hidden_dim, K = options.hidden_dim;

        // 1. Epilogue Functor 정의
        using EpilogueFunctor = cutlass::epilogue::thread::LinearCombination<
            Element, 8, float, float
        >;

        // 2. GemmLayernorm 템플릿 정의 (Stages=3 필수)
        using GemmFused = cutlass::GemmLayernorm<
            Element, LayoutA,               
            Element, LayoutB,               
            Element, LayoutC,               
            float,                          
            EpilogueFunctor,
            cutlass::gemm::GemmShape<128, 128, 32>, 
            cutlass::gemm::GemmShape<64, 64, 32>,   
            cutlass::gemm::GemmShape<16, 8, 16>,    
            3,                              // Stages0
            3,                              // Stages1
            false                           
        >;
        
        // --- V2: Fused Benchmark 내부 수정 ---
        GemmFused fused_op;

        // 🌟 FIX 1: LayerNorm 통계량 메모리 깔끔하게 초기화 (쓰레기값 방지)
        cudaMemset(tensor_mean.device_data(), 0, M * sizeof(Element));
        cudaMemset(tensor_variance.device_data(), 0, M * sizeof(Element));

        // 🌟 [핵심 추가] FIX 2: V2가 쓰레기값 C를 더하지 않도록 0으로 밀어버리기!
        cudaMemset(block_C.get(), 0, M * N * sizeof(Element));

        typename GemmFused::Arguments arguments(
            cutlass::gemm::GemmCoord{M, N, K},   // problem_size0
            cutlass::gemm::GemmCoord{M, N, K},   // 🌟 드디어 K로 복구! (두 번째 GEMM 정상 작동)

            block_A.get(),                       // ptr_A
            block_B.get(),                       // ptr_B
            block_C.get(),                       // ptr_C
            block_Bias.get(),                    // ptr_Bias
            block_Residual.get(),                // ptr_Residual
            
            // 🌟 포인터 순서도 B2B 풀코스에 맞게 원상 복구!
            block_D_intermediate.get(),          // ptr_D: 1차 GEMM 결과 (임시 버퍼)
            block_E_intermediate.get(),          // ptr_E: 2차 GEMM용 가중치 (입력2)
            block_D_v2.get(),                    // ptr_O: 2차 GEMM까지 마친 진짜 최종 출력!

            K,                                   // ldm_A
            K,                                   // ldm_B
            N,                                   // ldm_C
            0,                                   // ldm_Bias (Broadcast를 위해 0)
            N,                                   // ldm_Residual
            N,                                   // ldm_D
            N,                                   // ldm_E 
            N,                                   // ldm_O

            typename EpilogueFunctor::Params(options.alpha, options.beta), // linear_scaling

            cutlass::TensorRef<Element, cutlass::layout::RowMajor>(
                tensor_variance.device_data(), tensor_variance.layout()
            ),
            cutlass::TensorRef<Element, cutlass::layout::RowMajor>(
                tensor_mean.device_data(), tensor_mean.layout()
            ),
            cutlass::TensorRef<Element, cutlass::layout::RowMajor>(
                tensor_gamma.device_data(), tensor_gamma.layout()
            ),
            cutlass::TensorRef<Element, cutlass::layout::RowMajor>(
                tensor_beta.device_data(), tensor_beta.layout()
            ) //,

            //1e-6f // eps for LayerNorm  
        );

        cutlass::Status status = fused_op.initialize(arguments);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "Fused GEMM initialization failed! (Status: " << (int)status << ")" << std::endl;
            return 0.0f;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        // // Warmup
        // for(int i=0; i < options.warmup; ++i) fused_op();
        // cudaDeviceSynchronize();
        fused_op();
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR("V2: Fused Kernel (Warmup)");

        cudaEventRecord(start);
        for(int i=0; i < options.iterations; ++i) {
            fused_op();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        return elapsed_ms / options.iterations;
    }

    void run() {
        int M = options.batch_size * options.seq_len, N = options.hidden_dim, K = options.hidden_dim;
        double flops = 4.0 * M * N * K;
        double bytes_saved = (double)M * N * sizeof(Element) * 2.0;

        auto v1 = run_unfused();

        // V2를 실행하기 전에 V1의 1차 GEMM 결과를 미리 호스트로 백업
        std::vector<Element> host_v1_int(M * N);
        cutlass::device_memory::copy_to_host(host_v1_int.data(), block_D_intermediate.get(), M * N);

        // 버퍼를 0으로 초기화!
        cudaMemset(block_D_intermediate.get(), 0, M * N * sizeof(Element));

        float v2_ms = run_fused();

        float speedup = v1.total_ms / v2_ms;
        double tflops_v1 = (flops / (v1.total_ms * 1e-3)) / 1e12;
        double tflops_v2 = (flops / (v2_ms * 1e-3)) / 1e12;

        std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  B2B GEMM + Bias + Residual + LayerNorm Fusion Benchmark      ║\n";
        std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Problem: Batch=" << options.batch_size << ", Seq=" << options.seq_len
                  << " (M=" << M << "), N=" << N << ", K=" << K << std::string(15, ' ') << "║\n";
        std::cout << "║ GFLOPS: " << std::fixed << std::setprecision(2) << flops / 1e9 << std::string(46, ' ') << "║\n";
        std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ V1 (Unfused):                                                 ║\n";
        std::cout << "║   GEMM Only:         " << std::setw(8) << v1.gemm_ms << " ms                          ║\n";
        std::cout << "║   Post-Processing:   " << std::setw(8) << v1.post_ms << " ms                          ║\n";
        std::cout << "║   Total:             " << std::setw(8) << v1.total_ms << " ms (" << std::setprecision(2) << tflops_v1 << " TFLOPS)          ║\n";
        std::cout << "║                                                               ║\n";
        std::cout << "║ V2 (Fused):                                                   ║\n";
        std::cout << "║   Single Kernel:     " << std::setw(8) << v2_ms << " ms (" << std::setprecision(2) << tflops_v2 << " TFLOPS)          ║\n";
        std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Speedup: " << std::setprecision(2) << speedup << "x                                          ║\n";
        std::cout << "║ Memory Saved: " << std::setprecision(2) << bytes_saved / (1024*1024) << " MB (Intermediate C eliminated)         ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

        // 1. GPU 연산이 완전히 끝날 때까지 대기
        cudaDeviceSynchronize();

        // ---------------------------------------------------------
        // 🚨 디버깅: 1단계 (GEMM 1 결과) 확인
        // ---------------------------------------------------------
        //  이제 버퍼에는 V2의 결과만 남아있으므로 V2용 벡터에만 복사
        std::vector<Element> host_v2_int(M * N);
        cutlass::device_memory::copy_to_host(host_v2_int.data(), block_D_intermediate.get(), M * N);

        std::cout << "\n[단계 1: 1차 GEMM 결과물 확인]\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "V1 sample [0,0]: " << (float)host_v1_int[0] << " / V2 sample [0,0]: " << (float)host_v2_int[0] << "\n";
        // std::vector<Element> host_v1_int(M * N);
        // std::vector<Element> host_v2_int(M * N);
        // cutlass::device_memory::copy_to_host(host_v1_int.data(), block_D_intermediate.get(), M * N); 
        // cutlass::device_memory::copy_to_host(host_v2_int.data(), block_D_intermediate.get(), M * N);

        // std::cout << "\n[단계 1: 1차 GEMM 결과물 확인]\n";
        // std::cout << std::fixed << std::setprecision(6);
        // std::cout << "V1 sample [0,0]: " << (float)host_v1_int[0] << " / V2 sample [0,0] (버퍼 동일시): " << (float)host_v2_int[0] << "\n";

        // ---------------------------------------------------------
        // 🚨 디버깅: 2단계 (최종 GEMM 2 결과) 확인 및 오차 계산
        // ---------------------------------------------------------
        std::vector<Element> host_v1(M * N);
        std::vector<Element> host_v2(M * N);
        
        cutlass::device_memory::copy_to_host(host_v1.data(), block_D_v1_final.get(), M * N); 
        cutlass::device_memory::copy_to_host(host_v2.data(), block_D_v2.get(), M * N);
        
        std::cout << "\n[단계 3: 최종 GEMM 2 결과물 확인]\n";
        std::cout << "V1 sample [0,0]: " << (float)host_v1[0] << " / V2 sample [0,0]: " << (float)host_v2[0] << "\n";
        std::cout << "V1 sample [10,10]: " << (float)host_v1[10*N + 10] << " / V2 sample [10,10]: " << (float)host_v2[10*N + 10] << "\n";
        
        float max_diff = 0.0f;
        int nan_count = 0;
        for(int i = 0; i < M * N; ++i) {
            float v1_val = (float)host_v1[i];
            float v2_val = (float)host_v2[i];
            
            if (std::isnan(v2_val)) {
                nan_count++;
                continue;
            }

            float diff = std::abs(v1_val - v2_val);
            if (diff > max_diff) max_diff = diff;
        }

        if (nan_count > 0) {
            std::cout << "⚠️ Warning: Detected " << nan_count << " NaN values in V2 output!\n";
        }
        std::cout << "Max Absolute Error between V1 and V2: " << max_diff << "\n\n";
    }
};

int main(int argc, char const **argv) {
    Options options;
    options.parse(argc, argv);
    if (options.help) return 0;
    
    Benchmark bench(options);
    bench.run();
    return 0;
}