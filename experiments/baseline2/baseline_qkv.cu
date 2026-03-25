// /home/yh/Multimodal_on_Jetson/experiments/baseline2/baseline_qkv.cu
#include <cuda_runtime.h>
#include "baseline_qkv.cuh"

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

void launch_split_transpose_kernel(
    const cutlass::half_t* gemm_out, 
    const cutlass::half_t* bias,     
    cutlass::half_t* q_buf,
    cutlass::half_t* k_buf,
    cutlass::half_t* v_buf,
    int batch_size, int seq_len,
    int head_num, int kv_head_num, int size_per_head,
    cudaStream_t stream)
{
    int total_elements = batch_size * seq_len * (head_num + 2 * kv_head_num) * size_per_head;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    split_transpose_kernel<<<grid_size, block_size, 0, stream>>>(
        gemm_out, bias, q_buf, k_buf, v_buf, 
        batch_size, seq_len, head_num, kv_head_num, size_per_head
    );
}