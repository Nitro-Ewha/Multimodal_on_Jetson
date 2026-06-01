# Accelerating Vision Transformer Inference via Non-GEMM Kernel Fusion on Edge GPUs
<br/><br/>

## 🔑 Key Words

- Physical AI
- Multimodal Inference
- Edge GPU
- Vision Transformer
- Kernel Fusion
- CUTLASS
- TensorRT
- GPU Kernel Optimization
<br/>

## 👥 누구를 위해?

  엣지 GPU 환경에서 Vision Transformer 기반 멀티모달 모델의 추론 성능을 최적화하고자 하는 연구자 및 GPU 기반 AI 시스템 개발자를 대상으로 한다.<br/>
  특히 Jetson과 같은 제한된 하드웨어 환경에서 Vision-Language Model 또는 멀티모달 모델을 실행해야 하는 임베디드 AI 개발자, 그리고 Transformer block 내부의 kernel-level 병목을 분석하고 최적화하려는 시스템 연구자를 주요 타겟 사용자로 설정한다.
<br/><br/>

## 🎯 누구의 어떤 문제를 해결하기 위해?

최근 멀티모달 모델과 Vision-Language Model은 이미지와 텍스트를 함께 처리하기 위해 Vision Transformer 기반 구조를 널리 활용하고 있다. 그러나 모델 규모와 visual token 수가 증가하면서 연산량과 메모리 요구량이 함께 커지고 있으며, 엣지 GPU 환경에서는 이러한 부담이 더욱 크게 작용한다.<br/>
엣지 GPU는 서버급 GPU에 비해 연산 자원과 메모리 대역폭이 제한적이기 때문에, 단순히 GEMM 연산 속도만 높이는 것만으로는 전체 추론 latency를 충분히 줄이기 어렵다. 실제 Transformer block 내부에서는 GEMM뿐 아니라 QKV Split&Transpose, LayerNorm, Bias/Residual Add와 같은 비-GEMM 연산이 반복적으로 실행된다.<br/>
이러한 비-GEMM 연산은 계산량은 크지 않지만, 별도의 CUDA kernel로 실행되면서 kernel launch overhead를 발생시키고, 중간 활성화 텐서를 DRAM에 저장한 뒤 다시 읽어오는 과정을 반복한다. 그 결과, 연산량 대비 latency 기여도가 큰 memory-bound 병목이 발생한다.<br/>
TensorRT와 CUTLASS는 GEMM 및 일부 정형화된 연산 패턴에 대해 효과적인 최적화를 제공하지만, Vision Transformer 내부의 모든 비-GEMM 커널 병목을 자동으로 제거하지는 못한다. 따라서 본 프로젝트는 엣지 GPU 환경에서 멀티모달 Transformer 추론 시 발생하는 비-GEMM 커널의 launch overhead와 DRAM access를 주요 병목으로 정의하고, 이를 Kernel Fusion을 통해 구조적으로 줄이고자 한다.
<br/><br/>

## 🛠 어떤 기술을 사용해서?

본 프로젝트는 profiling 기반 병목 분석과 CUTLASS 기반 custom kernel 구현을 통해 Vision Encoder Block 내부의 비-GEMM 연산을 GEMM 실행 경로에 통합한다.
<br/><br/>

### 1. Profiling 기반 병목 분석

먼저 TensorRT 기반 baseline을 실행하고, Nsight Systems와 Nsight Compute를 활용하여 Transformer block 내부의 kernel execution 구조를 분석한다.<br/>
분석 대상은 다음과 같다.

- Kernel execution timeline
- Kernel launch overhead
- DRAM access
- Memory throughput
- Arithmetic intensity
- GPU utilization
- 비-GEMM kernel의 latency 비중

이를 통해 QKV Split&Transpose, LayerNorm, Bias/Residual Add와 같이 계산량은 작지만 memory traffic과 kernel launch overhead를 유발하는 연산을 주요 fusion 대상으로 선정한다.
<br/><br/>

### 2. QKV Split&Transpose Fusion

기존 attention 과정에서는 QKV projection GEMM 이후 Q, K, V를 분리하고 attention 연산에 맞는 layout으로 변환하기 위해 Split&Transpose kernel이 별도로 실행된다. 이 과정에서 GEMM output이 DRAM에 저장된 뒤 다시 읽히므로 불필요한 memory traffic이 발생한다.<br/>
본 프로젝트에서는 QKV projection GEMM의 epilogue 단계에서 split과 transpose를 동시에 수행하도록 fused kernel을 구현한다. GEMM accumulator register fragment를 바로 Q/K/V layout으로 변환하여 저장함으로써, 중간 결과를 DRAM에 materialize하지 않고 별도의 Split&Transpose kernel launch를 제거한다.<br/>
기대 효과는 다음과 같다.

- QKV Split&Transpose kernel 제거
- Intermediate tensor의 DRAM read/write 감소
- Kernel launch overhead 감소
- Sequence length 증가 시 fusion 효과 확대
<br/>

### 3. LayerNorm + Bias/Residual Add Fusion

기존 Transformer block에서는 GEMM 이후 Bias Add, Residual Add, LayerNorm이 독립적인 kernel로 실행된다. 이 구조는 intermediate activation을 DRAM에 반복적으로 저장하고 다시 읽게 하므로 memory traffic과 launch overhead를 증가시킨다.<br/>
본 프로젝트에서는 GEMM epilogue에서 Bias/Residual Add와 block-level statistics computation을 수행하고, lightweight reduction kernel을 통해 최종 mean과 variance를 계산한 뒤, 다음 GEMM stage에서 fused LayerNorm을 적용하는 구조를 사용한다.<br/>
이를 통해 standalone LayerNorm kernel을 줄이고, LayerNorm 결과를 별도의 intermediate tensor로 저장하지 않도록 실행 경로를 재구성한다.<br/>
기대 효과는 다음과 같다.

- Bias/Residual Add와 LayerNorm의 kernel overhead 감소
- Intermediate activation의 DRAM traffic 감소
- GEMM과 후처리 연산 간 데이터 이동 최소화
- Vision Encoder Block 전체 latency 감소
<br/>

## 🚀 무엇을 만들려고 하는가?

본 프로젝트는 엣지 GPU 환경에서 Vision Transformer 기반 멀티모달 모델의 추론을 가속하기 위한 Kernel Fusion 기반 최적화 시스템을 구축한다.<br/>
구체적으로는 다음을 구현하고 검증한다.

1. TensorRT baseline 기반 Vision Encoder Block profiling
2. Nsight Systems / Nsight Compute를 활용한 비-GEMM kernel 병목 분석
3. QKV Split&Transpose Fusion kernel 구현
4. LayerNorm + Bias/Residual Add Fusion kernel 구현
5. Baseline 대비 latency, DRAM access, speedup 비교
6. Fused kernel 적용 이후 output correctness 검증
7. Vision Encoder Block 수준에서의 end-to-end latency 감소 효과 분석

최종적으로 본 프로젝트는 엣지 GPU에서 멀티모달 Transformer 추론 시, 비-GEMM 커널의 kernel launch overhead와 DRAM access가 주요 병목으로 작용함을 보이고, GEMM 인접 비-GEMM 연산을 fused execution path로 흡수하는 것이 Vision Encoder Block 최적화의 핵심 방향임을 제안한다.
<br/><br/>

## 📌 기대 효과

- Kernel launch overhead 감소
- DRAM access 및 memory traffic 감소
- Vision Encoder Block latency 감소
- Memory-bound 비-GEMM 연산 병목 완화
- 엣지 GPU 환경에서 Vision Transformer 추론 효율 향상
- TensorRT baseline 이후에도 남는 kernel-level 병목에 대한 추가 최적화 가능성 제시
<br/>

## References

[1] A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," in ICML, 2021.
<br/>
[2] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.
<br/>
[3] A. Vaswani et al., "Attention is all you need," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017.
<br/>
[4] NVIDIA, “NVIDIA TensorRT Developer Guide,” NVIDIA Documentation.
<br/>
[5] A. Kerr et al., "CUTLASS: Fast Linear Algebra in CUDA C++, " NVIDIA Technical Blog, 2017.
<br/>
[6] NVIDIA, "CUTLASS 3.x Documentation," NVIDIA Documentation.
<br/>
[7] NVIDIA, “NVIDIA Jetson Orin Series,” NVIDIA Embedded Systems Documentation.
<br/>
[8] G. Bikshandi and J. Shah, “A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library,” arXiv preprint arXiv:2312.11918, 2023.
<br/>
[9] Zhao, Jie, et al. "Apollo: Automatic partition-based operator fusion through layer by layer optimization." Proceedings of Machine Learning and Systems 4 (2022): 1-19.
<br/>
[10] J. Li et al., “BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation,” in ICML, 2022.
<br/>
[11] M. Tschannen et al., “SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features,” arXiv preprint, 2025.
<br/><br/>
