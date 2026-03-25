// /home/yh/Multimodal_on_Jetson/experiments/baseline2/baseline_qkv.cuh
#pragma once

#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

// main 함수에서 부를 커널 런처 선언
void launch_split_transpose_kernel(
    const cutlass::half_t* gemm_out, 
    const cutlass::half_t* bias,     
    cutlass::half_t* q_buf,
    cutlass::half_t* k_buf,
    cutlass::half_t* v_buf,
    int batch_size, int seq_len,
    int head_num, int kv_head_num, int size_per_head,
    cudaStream_t stream = nullptr
);