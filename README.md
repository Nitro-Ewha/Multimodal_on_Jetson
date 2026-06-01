# Accelerating Vision Transformer Inference via Non-GEMM Kernel Fusion on Edge GPUs
<br/>

## 개요

최근 멀티모달 모델과 Vision-Language Model의 활용이 증가하면서, 이미지와 텍스트를 함께 처리하는 Vision Transformer 기반 구조가 널리 사용되고 있다. 그러나 Jetson과 같은 엣지 GPU 환경에서는 서버급 GPU에 비해 연산 자원과 메모리 대역폭이 제한적이기 때문에, Transformer 기반 모델의 실시간 추론에 어려움이 있다.<br/>
본 프로젝트는 멀티모달 모델의 Vision Encoder Block에서 발생하는 GPU kernel-level 병목을 분석하고, 비-GEMM 연산을 GEMM 인접 실행 경로에 흡수하는 Kernel Fusion 기법을 통해 추론 latency를 줄이는 것을 목표로 한다.<br/>
특히 QKV Split&Transpose, LayerNorm, Bias/Residual Add와 같은 비-GEMM 커널은 계산량은 크지 않지만, 별도의 CUDA kernel로 실행되며 kernel launch overhead와 DRAM access를 반복적으로 발생시킨다. 이러한 연산들은 TensorRT나 CUTLASS와 같은 기존 최적화 도구만으로는 완전히 제거되기 어렵기 때문에, 본 프로젝트에서는 CUTLASS 기반 custom kernel을 구현하여 해당 병목을 구조적으로 줄이고자 한다.
<br/><br/>

## 목표

본 프로젝트의 목표는 엣지 GPU 환경에서 Vision Transformer 추론 시 발생하는 비-GEMM 커널 병목을 분석하고, Kernel Fusion을 통해 latency와 memory traffic을 감소시키는 것이다.<br/>
주요 목표는 다음과 같다.

1. Vision Transformer 기반 멀티모달 모델의 kernel execution 구조 분석
2. Nsight Systems 및 Nsight Compute를 활용한 비-GEMM kernel 병목 식별
3. QKV Split&Transpose Fusion 구현
4. LayerNorm + Bias/Residual Add Fusion 구현
5. 기존 TensorRT baseline 대비 latency, DRAM access, speedup 비교
6. 엣지 GPU 환경에서의 Vision Encoder Block 최적화 가능성 검증
<br/>

## 문제 정의

최근 멀티모달 모델은 Vision Transformer 기반 구조를 활용하며, 모델 규모와 visual token 수 증가로 인해 연산량과 메모리 요구량이 크게 증가하고 있다. 그러나 엣지 GPU는 서버급 GPU에 비해 연산 자원과 메모리 대역폭이 제한적이어서, Transformer 추론 시 kernel launch overhead와 DRAM access가 latency 증가의 주요 원인이 된다.<br/>
특히 QKV Split&Transpose, LayerNorm, Bias/Residual Add와 같은 비-GEMM 연산은 계산량은 작지만 별도 CUDA kernel로 실행되며 반복적인 메모리 접근을 유발한다. TensorRT와 CUTLASS는 일부 연산 패턴에 대해 최적화를 제공하지만, 이러한 비-GEMM 커널 병목을 모두 제거하기에는 한계가 있다.<br/>
따라서 본 연구는 엣지 GPU 환경에서 발생하는 비-GEMM 커널의 launch overhead와 DRAM access를 주요 병목으로 정의하고, 이를 Kernel Fusion으로 줄이고자 한다.
<br/><br/>

## 제안 방법

본 프로젝트에서는 Vision Encoder Block 내부의 비-GEMM 연산을 GEMM 실행 경로에 통합하는 두 가지 Kernel Fusion 기법을 제안한다.

### 1. QKV Split&Transpose Fusion

기존 attention 과정에서는 QKV projection GEMM 이후, Q, K, V 텐서를 분리하고 attention 연산에 맞는 layout으로 변환하기 위해 Split&Transpose kernel이 별도로 실행된다. 이 연산은 계산량은 거의 없지만, 중간 텐서를 DRAM에 저장하고 다시 읽어오는 memory-bound 특성을 가진다.<br/>
본 프로젝트에서는 QKV projection GEMM의 epilogue 단계에서 split 및 transpose 연산을 함께 수행하도록 fused kernel을 구현하였다. 이를 통해 intermediate activation을 DRAM에 materialize하지 않고, GEMM accumulator register fragment에서 바로 Q/K/V layout으로 저장한다.<br/>
기대 효과는 다음과 같다.

- Split&Transpose kernel launch 제거
- Intermediate tensor의 DRAM read/write 감소
- Memory-bound transformation으로 인한 latency 감소
- Sequence length 증가 시 fusion 효과 확대
<br/>

### 2. LayerNorm + Bias/Residual Add Fusion

기존 Transformer block에서는 GEMM 이후 Bias Add, Residual Add, LayerNorm이 독립적인 kernel로 실행된다. 이 과정에서 intermediate activation이 DRAM에 반복적으로 저장되고 다시 읽히며, kernel launch overhead가 누적된다.<br/>
본 프로젝트에서는 GEMM epilogue에서 Bias/Residual Add와 block-level statistics computation을 수행하고, 이후 lightweight reduction kernel과 다음 GEMM stage를 연결하는 3-stage pipeline을 구성하였다. 이를 통해 standalone LayerNorm kernel을 줄이고, LayerNorm 결과를 다음 GEMM 실행 경로에 통합한다.<br/>
기대 효과는 다음과 같다.

- Bias/Residual Add kernel overhead 감소
- LayerNorm의 intermediate activation DRAM traffic 감소
- GEMM과 후처리 연산 간 데이터 이동 최소화
- Vision Encoder Block 전체 latency 감소
<br/>

## 시스템 구조

전체 최적화 과정은 다음과 같은 흐름으로 진행된다.

1. PyTorch 또는 TensorRT 기반 baseline 모델 실행
2. Nsight Systems를 활용한 kernel timeline 분석
3. Nsight Compute를 활용한 memory-bound kernel 식별
4. QKV Split&Transpose 및 LayerNorm + Bias/Residual Add를 fusion 대상으로 선정
5. CUTLASS 기반 custom fused kernel 구현
6. Baseline과 fused kernel의 latency, DRAM access, speedup, correctness 비교
7. Vision Encoder Block 수준에서 최종 latency 개선 효과 분석
<br/>

## 실험 및 평가 지표

본 프로젝트에서는 단순 실행 시간뿐 아니라, fusion을 통해 실제로 어떤 병목이 개선되었는지를 확인하기 위해 다음 지표를 사용한다.

- Latency
- Speedup
- DRAM access
- Output Correctness
<br/>

## 주요 결과

본 프로젝트에서는 BLIP Vision Encoder Block을 대상으로 비-GEMM 커널 퓨전의 효과를 분석하였다.
주요 결과는 다음과 같다.

- QKV Split&Transpose Fusion 이후 latency 감소
- Sequence length가 증가할수록 QKV Fusion 효과 증가
- LayerNorm + Bias/Residual Add Fusion을 통해 intermediate activation의 DRAM traffic 감소
- 기존 unfused 구조 대비 높은 speedup 달성
- TensorRT 기반 Vision Encoder Block에서 기존 kernel latency를 fused kernel latency로 치환한 결과, block-level latency 감소 확인

이를 통해 엣지 GPU 환경에서 Vision Transformer 추론 시 비-GEMM 커널의 launch overhead와 DRAM access가 주요 병목으로 작용하며, GEMM 인접 비-GEMM 연산을 fused execution path로 흡수하는 것이 Vision Encoder Block 최적화의 핵심 방향임을 확인하였다.
<br/><br/>

## 기술 스택

- C / C++
- CUDA
- CUTLASS
- Python
- PyTorch
- TensorRT
- NVIDIA Nsight Systems
- NVIDIA Nsight Compute
- GPU Server
- NVIDIA Jetson
<br/>

## 프로젝트 의의

본 프로젝트는 단순히 모델을 경량화하거나 구조를 변경하는 방식이 아니라, 기존 Vision Transformer 모델의 연산 의미와 출력은 유지하면서 GPU 내부 실행 방식을 최적화하는 시스템 수준의 연구이다.<br/>
특히 기존 최적화 도구가 충분히 처리하지 못하는 비-GEMM 커널 병목에 집중하여, kernel launch overhead와 DRAM access를 줄이는 방향으로 Transformer block의 실행 구조를 재설계하였다. 이를 통해 제한된 자원을 가진 엣지 GPU 환경에서도 멀티모달 모델의 추론 효율을 개선할 수 있음을 보인다.<br/>
<br/>
