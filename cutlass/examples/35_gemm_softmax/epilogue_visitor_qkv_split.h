/***************************************************************************************************
 * QKV Split + Transpose Epilogue Visitor for CUTLASS 2.x (Optimized v2)
 *
 * Key optimizations vs v1:
 *   1. visit() only buffers into fragment_D_ — NO stores
 *   2. end_step() does OutputVector (128-bit) vector stores, not per-element
 *   3. Address computation uses bit shifts (power-of-2 head_dim/hd)
 *   4. One address computation per vector, not per element
 *   5. Coalesced across threads: ThreadMap assigns consecutive N to adjacent threads,
 *      and consecutive N maps to consecutive D in [H, M, D] within the same head
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/numeric_conversion.h"

namespace cutlass {
namespace epilogue {
namespace threadblock {

template <
  typename ThreadblockShape_,
  int ThreadCount,
  typename OutputTileIterator_,
  typename ElementAccumulator_,
  typename ElementCompute_,
  typename ElementwiseFunctor_
>
class EpilogueVisitorQKVSplit {
public:

  using ThreadblockShape = ThreadblockShape_;
  static int const kThreadCount = ThreadCount;

  using OutputTileIterator = OutputTileIterator_;
  using ElementwiseFunctor = ElementwiseFunctor_;

  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  using ElementOutput = typename OutputTileIterator::Element;
  using LayoutOutput = cutlass::layout::RowMajor;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  // Required by GemmWithEpilogueVisitor kernel interface
  using ElementNorm = float;
  using ElementSum = float;

  using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
  using OutputVector = Array<ElementOutput, kElementsPerAccess>;

  static int const kThreadsPerRow = OutputTileIterator::ThreadMap::Detail::kAccessWidth;

  // Number of vectors per step (compile-time constant)
  static int const kVectorsPerStep =
    OutputTileIterator::Fragment::kElements / kElementsPerAccess;

  /// Argument structure
  struct Arguments {

    typename ElementwiseFunctor::Params   elementwise;
    int64_t                               batch_stride_C;
    int64_t                               batch_stride_D;
    int64_t                               batch_stride_Max;
    int64_t                               batch_stride_Sum;

    ElementOutput*                        ptr_Q;
    ElementOutput*                        ptr_K;
    ElementOutput*                        ptr_V;
    int                                   num_heads;
    int                                   head_dim;
    int                                   seq_len;
    int64_t                               batch_stride_QKV;

    Arguments():
      batch_stride_C(0), batch_stride_D(0),
      batch_stride_Max(0), batch_stride_Sum(0),
      ptr_Q(nullptr), ptr_K(nullptr), ptr_V(nullptr),
      num_heads(0), head_dim(0), seq_len(0), batch_stride_QKV(0)
    { }

    Arguments(
      typename ElementwiseFunctor::Params elementwise_,
      int64_t batch_stride_C_, int64_t batch_stride_D_,
      int64_t batch_stride_Max_, int64_t batch_stride_Sum_,
      ElementOutput* ptr_Q_, ElementOutput* ptr_K_, ElementOutput* ptr_V_,
      int num_heads_, int head_dim_, int seq_len_,
      int64_t batch_stride_QKV_ = 0
    ):
      elementwise(elementwise_),
      batch_stride_C(batch_stride_C_), batch_stride_D(batch_stride_D_),
      batch_stride_Max(batch_stride_Max_), batch_stride_Sum(batch_stride_Sum_),
      ptr_Q(ptr_Q_), ptr_K(ptr_K_), ptr_V(ptr_V_),
      num_heads(num_heads_), head_dim(head_dim_), seq_len(seq_len_),
      batch_stride_QKV(batch_stride_QKV_)
    { }
  };

  struct Params {
    typename ElementwiseFunctor::Params   elementwise;
    int64_t                               batch_stride_C;
    int64_t                               batch_stride_D;
    int64_t                               batch_stride_Max;
    int64_t                               batch_stride_Sum;

    ElementOutput*                        ptr_Q;
    ElementOutput*                        ptr_K;
    ElementOutput*                        ptr_V;
    int                                   num_heads;
    int                                   head_dim;
    int                                   seq_len;
    int                                   hd;             // num_heads * head_dim
    int64_t                               batch_stride_QKV;

    // Precomputed bit-shift constants
    int                                   head_dim_log2;
    int                                   hd_log2;
    bool                                  use_bitshift;

    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args):
      elementwise(args.elementwise),
      batch_stride_C(args.batch_stride_C), batch_stride_D(args.batch_stride_D),
      batch_stride_Max(args.batch_stride_Max), batch_stride_Sum(args.batch_stride_Sum),
      ptr_Q(args.ptr_Q), ptr_K(args.ptr_K), ptr_V(args.ptr_V),
      num_heads(args.num_heads), head_dim(args.head_dim),
      seq_len(args.seq_len),
      hd(args.num_heads * args.head_dim),
      batch_stride_QKV(args.batch_stride_QKV)
    {
      int hd_val = args.num_heads * args.head_dim;
      bool hd_pow2 = (hd_val & (hd_val - 1)) == 0 && hd_val > 0;
      bool d_pow2  = (args.head_dim & (args.head_dim - 1)) == 0 && args.head_dim > 0;
      use_bitshift = hd_pow2 && d_pow2;

      head_dim_log2 = 0;
      for (int v = args.head_dim; v > 1; v >>= 1) head_dim_log2++;
      hd_log2 = 0;
      for (int v = hd_val; v > 1; v >>= 1) hd_log2++;
    }
  };

  /// Shared storage
  struct SharedStorage { };

private:

  Params const &                        params_;
  SharedStorage &                       shared_storage_;
  MatrixCoord                           extent_;
  ElementwiseFunctor                    elementwise_;

  OutputTileIterator                    iterator_C_;
  OutputTileIterator                    iterator_D_;
  typename OutputTileIterator::Fragment fragment_C_;
  typename OutputTileIterator::Fragment fragment_D_;

  ElementAccumulator                    alpha_;
  ElementAccumulator                    beta_;

  ElementNorm                           *ptr_Max_;
  ElementSum                            *ptr_Sum_;

  int                                   batch_idx_;
  int64_t                               batch_offset_;

public:

  CUTLASS_DEVICE
  EpilogueVisitorQKVSplit(
    Params const &params,
    SharedStorage &shared_storage,
    cutlass::MatrixCoord const &problem_size,
    int thread_idx,
    int warp_idx,
    int lane_idx,
    typename OutputTileIterator::Params params_C,
    typename OutputTileIterator::Params params_D,
    typename OutputTileIterator::Element *ptr_C,
    typename OutputTileIterator::Element *ptr_D,
    ElementNorm *ptr_Max = nullptr,
    ElementSum *ptr_Sum = nullptr,
    cutlass::MatrixCoord const &threadblock_offset = cutlass::MatrixCoord(0, 0),
    int column_offset = 0
  ):
    params_(params),
    shared_storage_(shared_storage),
    extent_(problem_size),
    elementwise_(params.elementwise),
    iterator_C_(params_C, ptr_C, problem_size, thread_idx, threadblock_offset),
    iterator_D_(params_D, ptr_D, problem_size, thread_idx, threadblock_offset),
    ptr_Max_(ptr_Max),
    ptr_Sum_(ptr_Sum),
    batch_idx_(0),
    batch_offset_(0)
  {
    alpha_ = (params.elementwise.alpha_ptr ? *params.elementwise.alpha_ptr : params.elementwise.alpha);
    beta_  = (params.elementwise.beta_ptr  ? *params.elementwise.beta_ptr  : params.elementwise.beta);

    if (beta_ == ElementAccumulator()) {
      iterator_C_.clear_mask();
    }
  }

  CUTLASS_DEVICE
  void set_k_partition(int split_k_index, int split_k_slices) { }

  CUTLASS_DEVICE
  void set_batch_index(int batch_idx) {
    batch_idx_ = batch_idx;
    batch_offset_ = batch_idx * params_.batch_stride_QKV;
    iterator_C_.add_pointer_offset(batch_idx * params_.batch_stride_C);
    iterator_D_.add_pointer_offset(batch_idx * params_.batch_stride_D);
  }

  CUTLASS_DEVICE void begin_epilogue() { }

  CUTLASS_DEVICE
  void begin_step(int step_idx) {
    fragment_C_.clear();
    fragment_D_.clear();

    if (elementwise_.kScale != cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
      iterator_C_.load(fragment_C_);
      ++iterator_C_;
    }
  }

  CUTLASS_DEVICE void begin_row(int row_idx) { }

  //////////////////////////////////////////////////////////////////////////////
  /// visit(): compute and BUFFER only — no stores
  //////////////////////////////////////////////////////////////////////////////
  CUTLASS_DEVICE
  void visit(
    int iter_idx,
    int row_idx,
    int column_idx,
    int frag_idx,
    AccumulatorFragment const &accum) {

    OutputVector &source_vector = reinterpret_cast<OutputVector *>(&fragment_C_)[frag_idx];
    OutputVector &output = reinterpret_cast<OutputVector *>(&fragment_D_)[frag_idx];

    if (elementwise_.kScale == cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
      output = elementwise_(accum);
    } else {
      output = elementwise_(accum, source_vector);
    }
    // No store here — buffered in fragment_D_ for end_step()
  }

  CUTLASS_DEVICE void end_row(int row_idx) { }

  //////////////////////////////////////////////////////////////////////////////
  /// end_step(): vector stores with transposed addressing
  //
  /// Each iteration:
  ///   - One address computation per VecType (8 elements), not per element
  ///   - 128-bit store (same as VisitorAuxStore)
  ///   - Bit-shift addressing (no integer division)
  //////////////////////////////////////////////////////////////////////////////
  CUTLASS_DEVICE
  void end_step(int step_idx) {

    int const head_dim    = params_.head_dim;
    int const seq_len     = params_.seq_len;
    int const stride_head = seq_len * head_dim;

    MatrixCoord thread_start = iterator_D_.thread_start();

    CUTLASS_PRAGMA_UNROLL
    for (int frag_idx = 0; frag_idx < kVectorsPerStep; ++frag_idx) {

      MatrixCoord frag_coord = thread_start +
        OutputTileIterator::ThreadMap::iteration_offset(frag_idx);

      int m      = frag_coord.row();
      int n_base = frag_coord.column();

      bool guard = (m < extent_.row()) &&
                   (n_base + kElementsPerAccess - 1 < extent_.column());

      if (guard) {
        // ── One address computation per vector ──
        int qkv_idx, n_local, head_idx, dim_idx;

        if (params_.use_bitshift) {
          qkv_idx  = n_base >> params_.hd_log2;
          n_local  = n_base & (params_.hd - 1);
          head_idx = n_local >> params_.head_dim_log2;
          dim_idx  = n_local & (head_dim - 1);
        } else {
          qkv_idx  = n_base / params_.hd;
          n_local  = n_base - qkv_idx * params_.hd;
          head_idx = n_local / head_dim;
          dim_idx  = n_local - head_idx * head_dim;
        }

        // [H, M, D] row-major offset
        int out_offset = head_idx * stride_head + m * head_dim + dim_idx;

        ElementOutput *ptr_dst;
        switch (qkv_idx) {
          case 0: ptr_dst = params_.ptr_Q; break;
          case 1: ptr_dst = params_.ptr_K; break;
          default: ptr_dst = params_.ptr_V; break;
        }

        // ── 128-bit vector store ──
        OutputVector const &data =
          reinterpret_cast<OutputVector const *>(&fragment_D_)[frag_idx];

        arch::global_store<OutputVector, sizeof(OutputVector)>(
          data,
          (void *)(ptr_dst + out_offset + batch_offset_),
          true
        );
      }
    }

    ++iterator_D_;
  }

  CUTLASS_DEVICE void end_epilogue() { }
};

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

