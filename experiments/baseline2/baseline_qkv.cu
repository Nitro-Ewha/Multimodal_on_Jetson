// /home/yh/Multimodal_on_Jetson/experiments/baseline2/baseline_qkv.cu
#include <iostream>
#include <cuda_runtime.h>

// 공통 설정 파일 포함
#include "../common/common_config.h"

// ------------------------------------------------------------------
// 1. 별도의 Split + Transpose (+ Bias) CUDA 커널
// ------------------------------------------------------------------
__global__ void split_transpose_kernel(
    const cutlass::half_t* __restrict__ gemm_out, 
    const cutlass::half_t* __restrict__ bias,     
    cutlass::half_t* __restrict__ q_buf,
    cutlass::half_t* __restrict__ k_buf,
    cutlass::half_t* __restrict__ v_buf,
    int batch_size, int seq_len,
    int head_num, int kv_head_num, int size_per_head) 
{

    int N_dim = (head_num + 2 * kv_head_num) * size_per_head;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * N_dim;
    
    if (idx >= total_elements) return;

    int row_idx = idx / N_dim; 
    int col_idx = idx % N_dim; 

    cutlass::half_t val = gemm_out[idx] + bias[col_idx];

    int qkv_id, head_id, size_id;
    if (col_idx < head_num * size_per_head) {
        qkv_id = 0; // Q
        head_id = col_idx / size_per_head;
        size_id = col_idx % size_per_head;
    } else if (col_idx < (head_num + kv_head_num) * size_per_head) {
        qkv_id = 1; // K
        head_id = (col_idx - head_num * size_per_head) / size_per_head;
        size_id = col_idx % size_per_head;
    } else {
        qkv_id = 2; // V
        head_id = (col_idx - (head_num + kv_head_num) * size_per_head) / size_per_head;
        size_id = col_idx % size_per_head;
    }

    int batch_id = row_idx / seq_len;
    int seq_id = row_idx % seq_len;

    int kv_heads = (qkv_id == 0) ? head_num : kv_head_num;
    int target_idx = batch_id * (kv_heads * seq_len * size_per_head)
                   + head_id * (seq_len * size_per_head)
                   + seq_id * size_per_head
                   + size_id;

    if (qkv_id == 0)      q_buf[target_idx] = val;
    else if (qkv_id == 1) k_buf[target_idx] = val;
    else                  v_buf[target_idx] = val;
}

// ------------------------------------------------------------------
// 2. CUTLASS GEMM 타입 정의
// ------------------------------------------------------------------
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
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages // common_config에서 가져옴
>;

// ------------------------------------------------------------------
// 3. Main 함수
// ------------------------------------------------------------------
int main() {
    // common_config.h에서 가져온 상수를 사용
    size_t size_A = M_DIM * K_DIM * sizeof(cutlass::half_t);
    size_t size_B = K_DIM * N_DIM * sizeof(cutlass::half_t);
    size_t size_Out = M_DIM * N_DIM * sizeof(cutlass::half_t); 
    size_t size_Bias = N_DIM * sizeof(cutlass::half_t);
    
    size_t size_Q = BATCH_SIZE * HEAD_NUM * SEQ_LEN * SIZE_PER_HEAD * sizeof(cutlass::half_t);
    size_t size_K = BATCH_SIZE * KV_HEAD_NUM * SEQ_LEN * SIZE_PER_HEAD * sizeof(cutlass::half_t);
    size_t size_V = size_K;

    cutlass::half_t *d_A, *d_B, *d_Out, *d_Bias, *d_Q, *d_K, *d_V;

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_Out, size_Out);
    cudaMalloc(&d_Bias, size_Bias);
    cudaMalloc(&d_Q, size_Q);
    cudaMalloc(&d_K, size_K);
    cudaMalloc(&d_V, size_V);

    cudaMemset(d_A, 0, size_A);
    cudaMemset(d_B, 0, size_B);
    cudaMemset(d_Bias, 0, size_Bias);

    typename Gemm::Arguments arguments{
        {M_DIM, N_DIM, K_DIM},
        {d_A, K_DIM}, 
        {d_B, N_DIM}, 
        {d_Out, N_DIM}, 
        {d_Out, N_DIM}, 
        {1.0f, 0.0f} 
    };

    Gemm gemm_op;
    cutlass::Status status = gemm_op(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM Error: " << cutlassGetStatusString(status) << std::endl;
        return -1;
    }
    cudaDeviceSynchronize();

    int total_elements = M_DIM * N_DIM;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    split_transpose_kernel<<<grid_size, block_size>>>(
        d_Out, d_Bias, d_Q, d_K, d_V, 
        BATCH_SIZE, SEQ_LEN, HEAD_NUM, KV_HEAD_NUM, SIZE_PER_HEAD
    );
    cudaDeviceSynchronize();

    std::cout << "Baseline QKV Split+Transpose 성공적으로 완료!" << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_Out); cudaFree(d_Bias);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);

    return 0;
}