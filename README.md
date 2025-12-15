# 🤖 Optimized Kernel Fusion for Accelerating Multimodal Inference on Edge Device
<br/>

### 개요

최근 Vision-Language (VL) 및 멀티모달 모델의 발전으로 엣지 환경에서도 실시간 멀티모달 인식이 요구되고 있다. 그러나 임베디드 GPU는 연산 자원과 메모리가 제한적이며, 멀티모달 모델 실행 시 다중 커널 병렬 수행 기능이 없어, context switching overhead와 낮은 GPU utilization이 발생한다. 기존 연구들은 Knowledge Distillation, Pruning, Quantization 등 SW 수준의 모델 경량화에 초점을 맞췄지만, 실제 엣지 디바이스 상에서 HW-level 병목을 분석하고 커널 수준 최적화로 성능을 개선하는 접근은 부족하다.<br/>
따라서 본 연구는 Jetson의 하드웨어 구조적 특성을 고려해, profiling → kernel fusion → performance evaluation 단계를 통해 온디바이스 멀티모달 모델의 효율적 실행 전략을 제시하고자 한다. 


### 목표

1. 멀티모달 추론 모델의 커널 특성 분석 및 profiling
    a. Nsight Systems를 활용하여 각 커널의 SM 활용률, memory throughput 등을 분석
    b. Memory-bound / Compute-bound 연산 패턴 분류
2. 커널 퓨전 기반 하드웨어 수준 최적화 구현
    a. CUDA를 이용해 CNN과 Transformer 기반 연산 커널 직접 구현
    b. 다양한 fusion 전략 실험
    c. 커널 호출 횟수 감소 및 context switching overhead 완화
3. Jetson 환경에서의 성능 평가
    a. 기존 sequential execution 대비 latency, throughput, GPU utilization 비교
4. 온디바이스 멀티모달 추론의 최적화 시나리오 제시
    a. profiling 및 시뮬레이션 결과를 기반으로 Jetson 시리즈에 적합한 최적 커널 구성 및 실행 전략 제안
<br/>

### 기술 스택

`C / C++ / Python` <br/>
`PyTorch / TensorFlow` <br/>
`CUDA` <br/>
`NVIDIA Nsight Systems` <br/>
`NVIDIA Jetson` <br/>

