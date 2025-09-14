# Ideation
## 🔑 Key Words
**- NVIDIA Jetson**   
**- Multimodal**   
**- Optimization**   


## 👥 누구를 위해?
**Jetson Nano**와 같이 컴퓨팅 자원이 제한된 소형 임베디드 시스템에서, 계산 복잡도가 높은 **멀티모달(Multimodal) task**를 수행하고자 하는 연구원 및 개발자


## 🎯 누구의 어떤 문제를 해결하기 위해?
- 멀티모달 대형 모델의 등장은 AI 기술의 혁신을 불러일으켰으며, 이는 임베디드 및 엣지 디바이스로 확장되며 혁신적인 어플리케이션들이 개발될 수 있는 토대가 되어주었다.
- 이러한 물결 속에서 온디바이스 멀티모달 대형 모델의 수요가 급증하였으나, **하드웨어 리소스가 제한된 엣지 디바이스에서 효율적이고 정확한 추론을 달성하는 것은 여전히 중요한 연구 과제로 남아있다** [1].
- 여러 임베디드 시스템 중에서는 FPGA와 비교했을 때 GPU를 탑재하고 있어 더 높은 throughput을 보여 딥러닝 어플리케이션에 적합하고, CUDA 프로그래밍이 가능하여 모델을 쉽게 탑재할 수 있는 NVIDIA의 Jetson이 주목받고 있다. [2]
- 따라서 우리는 모델 경량화 및 추론 최적화와 같은 **SW Optimization**과, 프로파일링을 통해 멀티모달 연산의 병목을 분석하여 최적화된 하드웨어 아키텍처를 제안하는 **HW Optimization**과 같이 여러 접근 방법을 **Jetson Nano**에 적용시켜봄으로써, **온디바이스 멀티모달 대형 모델**의 퍼포먼스를 발전시키고자 한다. 


## 🛠 어떤 기술을 사용해서?

- **SW Optimization**
  - **모델 경량화 (Model Compression)**
    - Pruning: 중요하지 않은 weight 제거  
    - Quantization: FP32 → FP16 / INT8 변환 (특히 TensorRT INT8)  
    - Knowledge Distillation: 대형 모델 → 소형 모델 지식 전이  
  - **추론 최적화 (Inference Optimization)**
    - Graph Optimization (ONNX Runtime, TensorRT)  
    - Mixed Precision Inference (FP16 활용)  
  - **아키텍처 최적화 (Architecture-level)**
    - Efficient Architectures: MobileNet, EfficientNet 같은 경량 모델  
    - Fusion 전략 조정 (Early Fusion vs. Late Fusion) → modality 통합 연산 효율화

- **HW Optimization**
  - **프로파일러(ex: Nsight System)를 사용해 목표하는 task의 병목이 어디서 발생하는지 분석**
    - GPU 사용률, 메모리 사용량, Streaming Multiprocessor (SM) 사용률, 텐서 코어 사용량 등 [3]
    - 크게 세 가지 메트릭으로 나누어 측정할 수 있음
      1. **SoC 수준 메트릭**
        1) Throughput: 단위 시간 당 처리된 총 이미지 수
        2) Power: 와트 단위 전력 소비량
      2. **GPU 수준 메트릭**
        1) GPU Utilization: GPU 연산 시간
        2) GPU Memory: GPU 메모리 사용량
        3) SM 발생 사이클: Instruction이 발행된 SM의 사이클
        4) SM 활성 사이클: 최소 하나의 워프를 가지는 SM의 사이클
        5) TC (Tensor Core) Utilization: TC 활성 사이클/총 사이클 (%)
      3. **커널 수준 메트릭**
        1) Launch Stats: Kernel launch에 소요된 시간
        2) Sync Time: Kernel 동기화에 소요된 시간   
  - **시뮬레이터를 사용해 병목 분석 및 아키텍처 수정**
    - 시뮬레이터에서 자유롭게 Configuration 및 알고리즘을 수정하여, 시나리오 별 성능 비교 분석
      - Configuration에는 총 SM의 개수, 메모리 파티션의 개수, 공유 메모리의 크기, 메모리 접근 latency 등을 자유롭게 수정 가능
      - 새로운 아키텍처를 제안하게 될 경우, 시뮬레이터의 코드를 수정하여 기존 코드로 벤치마크를 돌렸을 때와의 성능 비교


## 🚀 무엇을 만들려고 하는가?
- Jetson 기반으로 동작하는 온디바이스 멀티모달 대형 모델  
- 멀티모달 모델에 경량화, 추론 최적화, 아키텍처 최적화 등 다양한 접근법을 사용하여 엣지 디바이스에서의 성능 비교 분석
- 멀티모달과 같이 복잡한 연산을 실행하였을 때 하드웨어에서의 병목 분석 및 그에 따른 새로운 아키텍처 제안 
- 최종적으로 **온디바이스 멀티모달 대형 모델**에서의 다양한 최적화 시나리오 비교 분석을 통해, 해당 분야에서의 SOTA를 달성하기 위한 인사이트를 제공하고자 한다.    


   
## References   
[1] Zhu Y, Lu H. Edge-side NPU inference optimization: Adaptation research of multimodal large models on qualcomm platforms. Intelligent Data Analysis. 2025;0(0). doi:10.1177/1088467X251342172   
[2] Mittal, S. (2019). A survey on optimized implementation of deep learning models on the nvidia jetson platform. Journal of Systems Architecture, 97, 428-442.   
[3] Chakraborty, A., Tavernier, W., Kourtis, A., Pickavet, M., Oikonomakis, A., & Colle, D. (2025). Profiling Concurrent Vision Inference Workloads on NVIDIA Jetson--Extended. arXiv preprint arXiv:2508.08430.   


