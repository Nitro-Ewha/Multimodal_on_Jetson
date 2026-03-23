// /home/yh/Multimodal_on_Jetson/experiments/common/common_config.h
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// ------------------------------------------------------------------
// 1. 워크로드 및 하드웨어 설정 (공통 하이퍼파라미터)
// ------------------------------------------------------------------
constexpr int BATCH_SIZE = 32;
constexpr int SEQ_LEN = 1024;
constexpr int HEAD_NUM = 12;
constexpr int KV_HEAD_NUM = 4;
constexpr int SIZE_PER_HEAD = 64;
constexpr int HIDDEN_DIM = 768;

// GEMM 크기 (M, N, K)
constexpr int M_DIM = BATCH_SIZE * SEQ_LEN;
constexpr int N_DIM = (HEAD_NUM + 2 * KV_HEAD_NUM) * SIZE_PER_HEAD;
constexpr int K_DIM = HIDDEN_DIM;

// ------------------------------------------------------------------
// 2. 데이터 타입 설정 (FP16 기준)
// ------------------------------------------------------------------
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// ------------------------------------------------------------------
// 3. CUTLASS GEMM 튜닝 파라미터 (RTX 3080 맞춤)
// ------------------------------------------------------------------
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

constexpr int NumStages = 3; // RTX 3080 메모리 초과 방지용
constexpr int Alignment = 8; // 메모리 접근 정렬 단위