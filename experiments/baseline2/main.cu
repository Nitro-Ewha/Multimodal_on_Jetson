// /home/yh/Multimodal_on_Jetson/experiments/baseline2/main.cu
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

// 공통 설정 및 헤더 포함
#include "../common/common_config.h"
#include "baseline_qkv.cuh"

// CUDA 에러 체크 매크로
#define CHECK_CUDA(x) do {                                        \
    cudaError_t err = (x);                                        \
    if (err != cudaSuccess) {                                     \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)    \
                  << " at line " << __LINE__ << std::endl;        \
        std::exit(1);                                             \
    }                                                             \
} while(0)

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, Alignment, ElementAccumulator, ElementAccumulator
>;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutA,
    ElementInputB, LayoutB,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages
>;

int main() {
    // -------------------------------------------------------------
    // 1. 메모리 크기 계산 (common_config.h 변수 사용!)
    // -------------------------------------------------------------
    size_t size_A = M_DIM * K_DIM * sizeof(cutlass::half_t);
    size_t size_B = K_DIM * N_DIM * sizeof(cutlass::half_t);
    size_t size_Out = M_DIM * N_DIM * sizeof(cutlass::half_t); 
    size_t size_Bias = N_DIM * sizeof(cutlass::half_t);
    
    size_t size_Q = BATCH_SIZE * HEAD_NUM * SEQ_LEN * SIZE_PER_HEAD * sizeof(cutlass::half_t);
    size_t size_K = BATCH_SIZE * KV_HEAD_NUM * SEQ_LEN * SIZE_PER_HEAD * sizeof(cutlass::half_t);
    size_t size_V = size_K;

    cutlass::half_t *d_A, *d_B, *d_Out, *d_Bias, *d_Q, *d_K, *d_V;

    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_Out, size_Out));
    CHECK_CUDA(cudaMalloc(&d_Bias, size_Bias));
    CHECK_CUDA(cudaMalloc(&d_Q, size_Q));
    CHECK_CUDA(cudaMalloc(&d_K, size_K));
    CHECK_CUDA(cudaMalloc(&d_V, size_V));

    CHECK_CUDA(cudaMemset(d_A, 0, size_A));
    CHECK_CUDA(cudaMemset(d_B, 0, size_B));
    CHECK_CUDA(cudaMemset(d_Bias, 0, size_Bias));

    // -------------------------------------------------------------
    // 2. CUTLASS GEMM 세팅
    // -------------------------------------------------------------
    typename Gemm::Arguments arguments{
        {M_DIM, N_DIM, K_DIM},
        {d_A, K_DIM}, {d_B, N_DIM}, 
        {d_Out, N_DIM}, {d_Out, N_DIM}, 
        {1.0f, 0.0f} 
    };

    Gemm gemm_op;

    // -------------------------------------------------------------
    // 3. Warm-up (GPU 예열 - 정확한 성능 측정을 위함)
    // -------------------------------------------------------------
    std::cout << "Warming up GPU..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        gemm_op(arguments);
        launch_split_transpose_kernel(d_Out, d_Bias, d_Q, d_K, d_V, 
                                      BATCH_SIZE, SEQ_LEN, HEAD_NUM, KV_HEAD_NUM, SIZE_PER_HEAD, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // -------------------------------------------------------------
    // 4. 프로파일링 (100번 돌려서 평균 시간 측정)
    // -------------------------------------------------------------
    cudaEvent_t start, mid, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&mid));
    CHECK_CUDA(cudaEventCreate(&end));

    int iters = 100;
    float gemm_ms_sum = 0.f, split_ms_sum = 0.f, total_ms_sum = 0.f;

    std::cout << "Profiling started for " << iters << " iterations..." << std::endl;

    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        
        // GEMM 실행
        cutlass::Status status = gemm_op(arguments);
        if (status != cutlass::Status::kSuccess) { std::cerr << "GEMM Error!\n"; return -1; }
        
        CHECK_CUDA(cudaEventRecord(mid));

        // Split & Transpose 실행
        launch_split_transpose_kernel(d_Out, d_Bias, d_Q, d_K, d_V, 
                                      BATCH_SIZE, SEQ_LEN, HEAD_NUM, KV_HEAD_NUM, SIZE_PER_HEAD, 0);
        
        CHECK_CUDA(cudaEventRecord(end));
        CHECK_CUDA(cudaEventSynchronize(end));

        float gemm_ms = 0.f, split_ms = 0.f, total_ms = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&gemm_ms, start, mid));
        CHECK_CUDA(cudaEventElapsedTime(&split_ms, mid, end));
        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, end));

        gemm_ms_sum += gemm_ms;
        split_ms_sum += split_ms;
        total_ms_sum += total_ms;
    }

    // -------------------------------------------------------------
    // 5. 결과 출력
    // -------------------------------------------------------------
    std::cout << "\n=== 🚀 Baseline Performance (Avg over " << iters << " iters) ===" << std::endl;
    std::cout << "1. GEMM Time          : " << (gemm_ms_sum / iters) << " ms\n";
    std::cout << "2. Split+Transpose    : " << (split_ms_sum / iters) << " ms\n";
    std::cout << "--------------------------------------------------\n";
    std::cout << ">> Total Latency      : " << (total_ms_sum / iters) << " ms\n";
    std::cout << "==================================================\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_Out); cudaFree(d_Bias);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);

    return 0;
}