cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(QKV_Baseline LANGUAGES CXX CUDA)

# C++ 및 CUDA 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 사용 중인 GPU 아키텍처 설정
set(CMAKE_CUDA_ARCHITECTURES "86")

# 실행 파일 이름(baseline_run)과 소스 코드(baseline_qkv.cu) 지정
add_executable(baseline_run main.cu baseline_qkv.cu)

# 상대 경로: experiments/baseline2 에서 3칸 밖으로 나가서 cutlass 찾기
target_include_directories(baseline_run PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../cutlass/tools/util/include
)