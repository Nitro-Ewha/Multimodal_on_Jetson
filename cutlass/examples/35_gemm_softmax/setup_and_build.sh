#!/bin/bash
#======================================================================
# GEMM + QKV Split/Transpose Fusion — 설치 및 빌드 스크립트
#
# 사용법:
#   1. 이 스크립트와 같은 디렉토리에 아래 파일들을 배치:
#      - epilogue_visitor_qkv_split.h
#      - gemm_with_qkv_split.h
#      - gemm_qkv_split.cu
#      - setup_and_build.sh (이 파일)
#
#   2. CUTLASS_DIR 환경변수 설정 (또는 ~/cutlass 에 clone):
#      export CUTLASS_DIR=~/cutlass
#
#   3. 실행:
#      chmod +x setup_and_build.sh
#      ./setup_and_build.sh
#======================================================================

set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

CUTLASS_DIR="${CUTLASS_DIR:-$HOME/cutlass}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$CUTLASS_DIR/examples/35_gemm_softmax"

# ── 1. 환경 체크 ──
[ -d "$CUTLASS_DIR" ] || fail "CUTLASS 디렉토리를 찾을 수 없습니다: $CUTLASS_DIR"
[ -f "$EXAMPLE_DIR/gemm_with_epilogue_visitor.h" ] || fail "Example 35를 찾을 수 없습니다"

# ── 2. 파일 복사 ──
info "QKV 파일을 example 35 디렉토리에 복사합니다..."

cp "$SCRIPT_DIR/epilogue_visitor_qkv_split.h" "$EXAMPLE_DIR/"
cp "$SCRIPT_DIR/gemm_with_qkv_split.h"        "$EXAMPLE_DIR/"
cp "$SCRIPT_DIR/gemm_qkv_split.cu"            "$EXAMPLE_DIR/"

ok "파일 복사 완료"

# ── 3. CMakeLists.txt 패치 ──
CMAKE_FILE="$EXAMPLE_DIR/CMakeLists.txt"

if grep -q "35_gemm_qkv_split" "$CMAKE_FILE"; then
  info "CMakeLists.txt는 이미 패치됨"
else
  info "CMakeLists.txt 패치 중..."
  cat >> "$CMAKE_FILE" <<'EOF'

cutlass_example_add_executable(
  35_gemm_qkv_split
  gemm_qkv_split.cu
  )
EOF
  ok "CMakeLists.txt 패치 완료"
fi

# ── 4. 빌드 ──
SM_VER="${SM_VER:-86}"
BUILD_DIR="$CUTLASS_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# CUDA 경로 감지
if [ -d "/usr/local/cuda/bin" ]; then
  CUDA_COMPILER="/usr/local/cuda/bin/nvcc"
elif command -v nvcc &>/dev/null; then
  CUDA_COMPILER="$(which nvcc)"
else
  # cuda-13.1 등 특정 버전 탐색
  CUDA_COMPILER=$(find /usr/local -name "nvcc" -path "*/bin/*" 2>/dev/null | sort -V | tail -1)
  [ -n "$CUDA_COMPILER" ] || fail "nvcc를 찾을 수 없습니다"
fi
info "CUDA compiler: $CUDA_COMPILER"

info "CMake 설정 (SM${SM_VER})..."
cmake "$CUTLASS_DIR" \
  -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
  -DCUTLASS_NVCC_ARCHS="${SM_VER}" \
  -DCMAKE_CUDA_ARCHITECTURES="${SM_VER}" \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_TOOLS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  2>&1 | tail -3

info "빌드 중 (35_gemm_qkv_split)..."
cmake --build . --target 35_gemm_qkv_split -j$(nproc) 2>&1

BINARY="./examples/35_gemm_softmax/35_gemm_qkv_split"
[ -f "$BINARY" ] || fail "빌드 실패"
ok "빌드 완료: $BINARY"

# ── 5. 실행 ──
echo ""
info "========================================="
info " 실행: 기본 테스트 (seq=128, hidden=256, heads=4, dim=64)"
info "========================================="
echo ""

$BINARY --seq_len=128 --hidden=256 --num_heads=4 --head_dim=64 --batch_count=1 --iterations=10

echo ""
info "========================================="
info " 실행: 대형 테스트 (seq=512, hidden=1024, heads=8, dim=128)"
info "========================================="
echo ""

$BINARY --seq_len=512 --hidden=1024 --num_heads=8 --head_dim=128 --batch_count=2 --iterations=10

echo ""
ok "모든 테스트 완료!"
