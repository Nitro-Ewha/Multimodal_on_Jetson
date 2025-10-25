## 🔑 Key Words
**- NVIDIA Jetson**   
**- Multimodal**   
**- Kernel Fusion** 
**- Embedded GPU Optimization**   


## 👥 누구를 위해?
**Jetson Nano**와 같이 컴퓨팅 자원이 제한된 소형 임베디드 시스템에서 계산 복잡도가 높은 **멀티모달(Multimodal) 모델의 실시간 추론**를 수행하고자 하는 엣지 AI 연구원 및 개발자


## 🎯 누구의 어떤 문제를 해결하기 위해?
- 멀티모달 대형 모델의 등장은 AI 기술의 혁신을 불러일으켰으며, 이는 임베디드 및 엣지 디바이스로 확장되며 혁신적인 어플리케이션들이 개발될 수 있는 토대가 되어주었다.
- 이러한 물결 속에서 온디바이스 멀티모달 대형 모델의 수요가 급증하였으나, 하드웨어 리소스가 제한된 엣지 디바이스에서 효율적이고 정확한 추론을 달성하는 것은 여전히 중요한 연구 과제로 남아있다 [1].
- 여러 임베디드 시스템 중에서는 FPGA와 비교했을 때 GPU를 탑재하고 있어 더 높은 throughput을 보여 딥러닝 어플리케이션에 적합하고, CUDA 프로그래밍이 가능하여 모델을 쉽게 탑재할 수 있는 NVIDIA의 Jetson이 주목받고 있다 [2].
- 하지만 Jetson과 같은 임베디드 GPU는 연산 자원이 제한적이며, 다중 커널 병렬 실행을 지원하지 않아, 멀티모달 모델의 여러 인코더 커널을 동시에 실행할 수 없다.
- 그 결과, 각 모달리티의 인코더가 순차적으로 실행되어 context switching overhead와 낮은 GPU utilization이 발생하며 전체 추론 latency가 증가한다.
- 따라서 임베디드 GPU 환경에서의 멀티모달 추론 병목을 완화하기 위해 CNN과 Transformer 기반 커널을 하나로 융합하는 커널 퓨전 기법을 제안한다.


## 🛠 어떤 기술을 사용해서?

**문제 정의**
- CNN 기반 Vision Encoder → Compute-bound
- Transformer 기반 Text Encoder → Memory-bound
- 두 인코더를 독립적으로 실행 시 GPU 리소스가 불균형하게 사용됨
  → 일부 SM(Stream Multiprocessor) 또는 Memory Controller 유휴 상태 발생

**제안 기법**

- 각 인코더 커널의 연산 특성을 프로파일링하여 서로 보완적인 (complementary) 자원 사용 패턴을 가진 커널 쌍 선정
- 두 커널을 단일 fused kernel로 통합하여 실행
  → 커널 호출 횟수 감소 + context switching overhead 최소화
  → GPU utilization 극대화 및 latency 단축

**구현 방식**

- CUDA 기반 커널 단위 구현
    - Intra-thread Fusion
    - Intra-block Fusion
    - Inter-block Fusion
      → thread/block 계층별 자원 분배 전략 비교
        
- Profiling Metrics (Night System)
    - GPU utilization, SM occupancy, DRAM throughput
    - Kernel launch, synchronization time
- 비교 대상: 기존 sequential execution vs fused execution

## 🚀 무엇을 만들려고 하는가?
- - Jetson 기반으로 동작하는 온디바이스 멀티모달 대형 모델
- 멀티모달 모델에 경량화, 추론 최적화, 아키텍처 최적화 등 다양한 접근법을 사용하여 엣지 디바이스에서의 성능 비교 분석
- 멀티모달과 같이 복잡한 연산을 실행하였을 때 하드웨어에서의 병목 분석 및 그에 따른 새로운 아키텍처 제안
- 최종적으로 **온디바이스 멀티모달 대형 모델**에서의 다양한 최적화 시나리오 비교 분석을 통해, 해당 분야에서의 SOTA를 달성하기 위한 인사이트를 제공

   
## References   
[1] Zhu Y, Lu H. Edge-side NPU inference optimization: Adaptation research of multimodal large models on qualcomm platforms. Intelligent Data Analysis. 2025;0(0). doi:10.1177/1088467X251342172   
[2] Mittal, S. (2019). A survey on optimized implementation of deep learning models on the nvidia jetson platform. Journal of Systems Architecture, 97, 428-442.   
[3] Chakraborty, A., Tavernier, W., Kourtis, A., Pickavet, M., Oikonomakis, A., & Colle, D. (2025). Profiling Concurrent Vision Inference Workloads on NVIDIA Jetson--Extended. arXiv preprint arXiv:2508.08430.   


