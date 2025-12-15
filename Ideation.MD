최근 Vision-Language (VL) 및 멀티모달 모델의 발전으로 엣지 환경에서도 실시간 멀티모달 인식이 요구되고 있다. 그러나 임베디드 GPU는 연산 자원과 메모리가 제한적이며, 멀티모달 모델 실행 시 다중 커널 병렬 수행 기능이 없어, context switching overhead와 낮은 GPU utilization이 발생한다. 기존 연구들은 Knowledge Distillation, Pruning, Quantization 등 SW 수준의 모델 경량화에 초점을 맞췄지만, 실제 엣지 디바이스 상에서 HW-level 병목을 분석하고 커널 수준 최적화로 성능을 개선하는 접근은 부족하다.
    따라서 본 연구는 Jetson의 하드웨어 구조적 특성을 고려해, profiling → kernel fusion → performance evaluation 단계를 통해 온디바이스 멀티모달 모델의 효율적 실행 전략을 제시하고자 한다.   


## 👥 누구를 위해?
- Edge GPU 환경(e.g. Jetson)에서 VLM을 활용하는 연구자 및 ML 엔지니어
- 실시간 인식 시스템(예: 로봇, AR/VR, 스마트 카메라) 개발자
- 제한된 전력/자원에서 저지연 Vision-Language AI가 필요한 서비스 운영자
→ Dual Encoder 기반 VLM을 사용하는 연구자 및 개발자



## 🎯 누구의 어떤 문제를 해결하기 위해?
Dual-Encoder 기반 VLM을 사용하는 연구자 및 개발자들은 두 Encoder가  독립적으로 동작함에도 불구하고 실제 inference 과정에서 sequential하게 실행되어 GPU 자원이 비효율적으로 소모되는 문제를 겪고 있다. 특히 Edge GPU (Jetson Nano/Orin 등)에서는 다음 문제가 심각해진다:
- GPU compute 자원의 idle time 증가
- inference latency 증가로 실시간성 저하
- 전력 대비 성능 효율 감소 → 배터리 기반 장비에서 큰 문제
    본 프로젝트는 두 Encoder가 병렬로 실행되도록 구조를 재설계하고, kernel fusion을 적용하여 불필요한 overhead를 줄임으로써 GPU utilization을 극대화하고 inference latency를 효과적으로 줄이는 것을 목표로 한다.



## 🛠 어떤 기술을 사용해서?
- GPU Kernel Fusion을 통한 launch overhead 및 메모리 패스 축소
- Concurrent Kernel Execution 및 CUDA Streams 기반 병렬 스케줄링
- Nsight Systems / Compute 기반 profiling → 병목 구간 최적화
- CLIP 및 SigLIP2 Encoder 기반 Edge GPU 실제 실험
<br/>

**문제 정의** <br/>
Dual-Encoder 구조의 Vision-Language Model은 구조적으로 Vision Encoder와 Text Encoder가 병렬 실행이 가능함에도 불구하고, 실제 구현에서는 sequential execution이 발생해 GPU 자원 활용률이 낮아지고 전체 inference 시간이 증가하는 문제가 존재한다.
본 프로젝트는 kernel fusion과 병렬 실행 스케줄링을 통해 GPU utilization을 극대화하고 inference latency를 감소시키는 것을 목표로 한다.

<br/>

**제안 기법**
1. **Dual Encoder 병렬 실행 최적화**
    1. Vision Encoder와 Text Encoder의 연산 dependency를 분석
    2. 병렬 실행 가능한 연산 경로를 scheduling하여 두 Encoder가 동시에 GPU 자원을 사용할 수 있도록 구성
2. **Kernel Fusion 기반 연산 효율 개선**
    1. Encoder 내부의 반복적/연속적 연산을 fused kernel로 통합
    2. Memory traffic 감소 + launch overhead 최소화
    3. Edge 환경에서 효율성 극대화
3. **Overall GPU Utilization 향상**
    1. 병렬 실행 + kernel fusion 조합
    2. 실제 GPU active cycle을 증가시키고 idle time을 최소화
<br/>

**구현 방식**
1. Baseline Dual-Encoder VLM 모델 선정
2. Profiling
    1. Nsight Systems & Nsight Compute로 두 Encoder의 실행 타임라인 분석
    2. Sequential execution 구간 및 idle time 파악
3. 병렬 스케줄링 설계 및 검증
4. Encoder 내부 연산에 대한 custom fused kernel prototype 구현
5. 병렬 실행 + kernel fusion 조합 → Edge GPU 환경에서 최적화 적용
6. 성능 측정 (Latency, Utilization, Throughput)
7. Baseline 대비 성능 개선 평가

## 🚀 무엇을 만들려고 하는가?
Dual-Encoder VLM(CLIP/Siglip2)의 Vision/Text encoder를 Edge GPU 상에서 병렬화 및 kernel fusion을 적용하여 inference latency를 줄이는 최적화 시스템을 구현하고자 한다.

   
## References   
[1] Zhu Y, Lu H. Edge-side NPU inference optimization: Adaptation research of multimodal large models on qualcomm platforms. Intelligent Data Analysis. 2025;0(0). doi:10.1177/1088467X251342172   
[2] Mittal, S. (2019). A survey on optimized implementation of deep learning models on the nvidia jetson platform. Journal of Systems Architecture, 97, 428-442.   
[3] Chakraborty, A., Tavernier, W., Kourtis, A., Pickavet, M., Oikonomakis, A., & Colle, D. (2025). Profiling Concurrent Vision Inference Workloads on NVIDIA Jetson--Extended. arXiv preprint arXiv:2508.08430.   


