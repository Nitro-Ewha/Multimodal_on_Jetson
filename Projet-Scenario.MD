### Team 29 Nitro

| 항목 | 내용 |
| --- | --- |
| 프로젝트명 | **Optimized Kernel Fusion for Accelerating Multimodal Inference on Edge Devices** |
| 프로젝트 키워드 | NVIDIA Jetson, Multimodal, CUTLASS, Kernel Fusion, Embedded GPU Optimization |
| 트랙 | 연구 |
| 프로젝트 멤버 | 박세진, 박윤형, 강선화 |
| 팀 지도교수 | 윤명국 교수님 |
| 무엇을 만들고자 하는가 | Jetson Nano와 같은 임베디드 GPU 환경에서 실행되는 멀티모달 모델의 추론 과정 중 발생하는 GPU kernel-level 병목을 분석하고 이를 Kernel Fusion 및 실행 구조 최적화를 통해 GPU utilization을 향상시키는 시스템을 구축한다. |
| 고객 | 멀티모달 모델의 inference 성능 최적화에 관심 있는 연구자 및 GPU 기반 AI 시스템 개발자 |
| Pain Point | 멀티모달 모델의 크기와 연산량이 빠르게 증가하면서, Jetson과 같은 엣지 디바이스에서 실시간 추론을 수행하는 것이 점점 어려워지고 있다. 특히 엣지 환경은 서버급 GPU에 비해 연산 자원과 메모리 대역폭이 제한적이기 때문에, kernel launch overhead, 불필요한 DRAM access, 낮은 GPU utilization과 같은 비효율이 더욱 크게 성능 병목으로 이어진다. 이러한 문제는 end-to-end latency를 증가시켜 실시간 서비스 구현을 어렵게 만든다. <br/>TensorRT와 같은 기존 최적화 프레임워크는 효과적인 범용 커널 퓨전을 제공하지만, general-purpose 방식이기 때문에 멀티모달 모델 구조에서 발생하는 추가적인 fusion opportunity를 충분히 활용하지 못하는 한계가 있다. <br/>따라서 엣지 디바이스에 적합한 멀티모달 모델 최적화를 위해, profiling 기반으로 fusion 가능한 커널을 체계적으로 분석하고 최적의 커널 퓨전 조합을 찾는 연구가 필요하다. |
| 사용할 소프트웨어 패키지의 명칭과 핵심기능/용도 | 1. 멀티모달 모델 프레임워크 <br/>- PyTorch: 멀티모달 모델 로딩 및 추론 <br/>- CUDA/cuDNN: 가속 연산 및 커널 코드 구현 <br/><br/>2. 커널 최적화 및 경량화 툴 <br/>- TensorRT: 모델 경량화 및 inference graph 최적화 <br/>- ONNX / ONNX Runtime: 모델 변환 및 다양한 하드웨어에서 추론 가능 <br/>- CUTLASS: GEMM kernel 및 epilogue fusion 구현 <br/>- TensorRT Plugin: fused kernel을 inference graph에 통합 <br/><br/>3. Profiling & 분석 도구 <br/>- Nsight Systems: GPU/CPU layer 별 latency 및 kernel-level utilization 분석 <br/>- Nsight Compute: Kernel level utilization 분석 <br/>- PyTorch Profiler: 연산별 latency와 memory 사용량 분석 <br/>- TorchMetrics: multimodal 성능 평가 |
| 사용할 소프트웨어 패키지의 명칭과 URL | - PyTorch: https://pytorch.org <br/>- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit <br/>- TensorRT: https://developer.nvidia.com/tensorrt <br/>- ONNX Runtime: https://onnxruntime.ai <br/>- CUTLASS: https://github.com/NVIDIA/cutlass <br/>- TensorRT Plugin: https://github.com/NVIDIA/TensorRT/tree/main/plugin <br/>- Nsight Systems: https://developer.nvidia.com/nsight-systems <br/>- Nsight Compute: https://developer.nvidia.com/nsight-compute  <br/>- TorchMetrics: https://lightning.ai/docs/torchmetrics/stable/ |
| 팀그라운드룰 URL | https://github.com/Nitro-Ewha/Multimodal_on_Jetson/blob/main/GroudRule.MD |
| 최종수정일 | 2026.03.26. |
