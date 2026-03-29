/***************************************************************************************************
 * GemmQKVSplit: GEMM + QKV Split/Transpose fusion
 *
 * Composes:
 *   - DefaultGemm kernel (mainloop)
 *   - EpilogueVisitorQKVSplit (epilogue visitor)
 *   - GemmWithEpilogueVisitor (kernel dispatch from example 35)
 *
 * Unlike GemmSoftmax, this does NOT need a second-pass reduction kernel.
 * Everything is fused into a single kernel launch.
 *
 **************************************************************************************************/

#pragma once

#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"

#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

#include "epilogue_visitor_qkv_split.h"

// Reuse the GemmWithEpilogueVisitor kernel from example 35
#include "gemm_with_epilogue_visitor.h"

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename LayoutB_,
  typename ElementC_,
  typename ElementCompute_,
  typename OperatorClass_,
  typename ArchTag_,
  typename ThreadblockShape_,
  typename WarpShape_,
  typename InstructionShape_,
  typename EpilogueFunctorOp_,
  int kStages_,
  int AlignmentA_ = 128 / cutlass::sizeof_bits<ElementA_>::value,
  int AlignmentB_ = 128 / cutlass::sizeof_bits<ElementB_>::value
>
class GemmQKVSplit {
public:

  //
  // Type definitions
  //

  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;
  using ElementCompute = ElementCompute_;

  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;

  using EpilogueFunctorOp = EpilogueFunctorOp_;

  using LayoutC = cutlass::layout::RowMajor;

  using TensorRefA = TensorRef<ElementA, LayoutA>;
  using TensorRefB = TensorRef<ElementB, LayoutB>;
  using TensorRefC = TensorRef<ElementC, LayoutC>;

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape        = WarpShape_;
  using InstructionShape = InstructionShape_;

  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;

  static int const kStages    = kStages_;
  static int const AlignmentA = AlignmentA_;
  static int const AlignmentB = AlignmentB_;

  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // Basic GEMM kernel (provides Mma, Epilogue types)
  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementC, LayoutC, ElementCompute,
    OperatorClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueFunctorOp,
    ThreadblockSwizzle,
    kStages,
    true,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass, ArchTag, ElementA, ElementB, ElementC, ElementCompute>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone
  >::GemmKernel;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // Epilogue visitor
  using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorQKVSplit<
    ThreadblockShape,
    DefaultGemmKernel::kThreadCount,
    typename DefaultGemmKernel::Epilogue::OutputTileIterator,
    ElementCompute,     // ElementAccumulator (float) — matches GEMM accumulators
    ElementCompute,     // ElementCompute (float)
    EpilogueFunctorOp
  >;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    EpilogueVisitor,
    typename DefaultGemmKernel::Epilogue
  >::Epilogue;

  // GEMM kernel with visitor
  using GemmKernel = gemm::kernel::GemmWithEpilogueVisitor<
    typename DefaultGemmKernel::Mma,
    Epilogue,
    ThreadblockSwizzle
  >;

public:

  /// Arguments class
  struct Arguments {

    typename GemmKernel::Arguments gemm;
    cutlass::gemm::GemmCoord extend;

    Arguments() { }

    Arguments(
      cutlass::gemm::GemmCoord problem_size,
      int32_t batch_count_,
      TensorRefA ref_A_,
      TensorRefB ref_B_,
      TensorRefC ref_C_,       // source matrix (for beta*C, can be dummy if beta=0)
      TensorRefC ref_D_,       // dummy output (not actually written; QKV goes to visitor)
      typename EpilogueFunctorOp::Params linear_scaling,
      ElementC* ptr_Q_,
      ElementC* ptr_K_,
      ElementC* ptr_V_,
      int num_heads_,
      int head_dim_,
      int64_t batch_stride_A_ = 0,
      int64_t batch_stride_B_ = 0,
      int64_t batch_stride_C_ = 0,
      int64_t batch_stride_D_ = 0,
      int64_t batch_stride_QKV_ = 0
    ):
      gemm(
        cutlass::gemm::GemmUniversalMode::kBatched,
        problem_size,
        batch_count_,
        ref_A_,
        ref_B_,
        ref_C_,
        ref_D_,
        nullptr,   // ptr_Max (unused)
        nullptr,   // ptr_Sum (unused)
        batch_stride_A_,
        batch_stride_B_,
        typename EpilogueVisitor::Arguments(
          linear_scaling,
          batch_stride_C_,
          batch_stride_D_,
          0,  // batch_stride_Max (unused)
          0,  // batch_stride_Sum (unused)
          ptr_Q_,
          ptr_K_,
          ptr_V_,
          num_heads_,
          head_dim_,
          problem_size.m(),  // seq_len = M
          batch_stride_QKV_
        )
      ),
      extend(problem_size)
    { }
  };

  struct Params {
    typename GemmKernel::Params gemm;
    MatrixCoord extend;

    Params() { }

    Params(Arguments const &args):
      gemm(args.gemm),
      extend(MatrixCoord(args.extend.m(), args.extend.n()))
    { }
  };

private:
  Params params_;

public:

  GemmQKVSplit() { }

  Status initialize(Arguments const &args) {
    params_ = Params(args);
    return cutlass::Status::kSuccess;
  }

  Status run(cudaStream_t stream = nullptr) {

    dim3 gemm_grid = ThreadblockSwizzle().get_grid_shape(params_.gemm.grid_tiled_shape);
    dim3 gemm_block(GemmKernel::kThreadCount, 1, 1);

    int gemm_smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    cudaError_t result;

    if (gemm_smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(
        cutlass::Kernel<GemmKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        gemm_smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<GemmKernel><<<gemm_grid, gemm_block, gemm_smem_size, stream>>>(params_.gemm);

    result = cudaGetLastError();
    if (result != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(result) << std::endl;
      return cutlass::Status::kErrorInternal;
    }

    return cutlass::Status::kSuccess;
  }

  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
