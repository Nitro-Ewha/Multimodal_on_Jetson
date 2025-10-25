# 🤖 Optimized Kernel Fusion for Accelerating Multimodal Inference on Edge Device
<br/>

### 개요

최근 멀티모달 대형 모델의 발전으로 텍스트, 이미지, 음성 등 다양한 센서 데이터를 통합 처리하는 AI 응용이 빠르게 확산되고 있다. <br/>
하지만 Jetson과 같은 임베디드 엣지 디바이스에서는 제한된 연산 자원과 메모리로 인해 이러한 대형 모델을 효율적으로 실행하기 어렵다. 특히, 멀티모달 모델의 여러 인코더 커널이 순차적으로 실행되면서 context switching overhead가 빈번히 발생하고 GPU utilization이 저하되는 현상이 발생한다. <br/>
본 프로젝트는 이러한 문제를 해결하기 위해, CNN (Compute-bound) 과 Transformer (Memory-bound) 커널을 하나로 결합하는 커널 퓨전 (Kernel Fusion) 기반의 최적화 기법을 제안하고 구현한다. 이를 통해, Jetson 환경에서의 멀티모달 추론 시 GPU 자원 활용률 향상, latency 단축, throughput 개선을 달성하는 것을 목표로 한다.
<br/>

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

