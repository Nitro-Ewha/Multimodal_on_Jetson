### Team 29 Nitro

| 항목 | 내용 |
| --- | --- |
| 프로젝트명 | **Kernel-Fused Parallel Execution for Accelerating Dual-Encoder Vision-Language Models** |
| 프로젝트 키워드 | NVIDIA Jetson, Multimodal, Kernel Fusion, Embedded GPU Optimization |
| 트랙 | 연구 |
| 프로젝트 멤버 | 박세진, 박윤형, 강선화 |
| 팀 지도교수 | 윤명국 교수님 |
| 무엇을 만들고자 하는가 | Jetson Nano와 같은 임베디드 GPU 환경에서 실행되는 멀티모달 모델의 추론 과정 중 발생하는 GPU 리소스 병목을 분석하고 이를 Kernel Fusion 기법을 통해 최적화하는 시스템을 구축한다. |
| 고객 | 컴퓨팅 자원이 제한된 소형 임베디드 GPU 환경에서 실시간 멀티모달 추론을 수행하고자 하는 연구원 및 개발자 |
| Pain Point | 최근 Vision-Language (VL) 및 멀티모달 모델의 발전으로 엣지 환경에서도 실시간 멀티모달 인식이 요구되고 있다. 그러나 임베디드 GPU는 연산 자원과 메모리가 제한적이며, 멀티모달 모델 실행 시 다중 커널 병렬 수행 기능이 없어, context switching overhead와 낮은 GPU utilization이 발생한다. <br/>기존 연구들은 Knowledge Distillation, Pruning, Quantization 등 SW 수준의 모델 경량화에 초점을 맞췄지만, 실제 엣지 디바이스 상에서 HW-level 병목을 분석하고 커널 수준 최적화로 성능을 개선하는 접근은 부족하다. <br/> 따라서 본 연구는 Jetson의 하드웨어 구조적 특성을 고려해, profiling → kernel fusion → performance evaluation 단계를 통해 온디바이스 멀티모달 모델의 효율적 실행 전략을 제시하고자 한다. |
| 사용할 소프트웨어 패키지의 명칭과 핵심기능/용도 | 1. 멀티모달 모델 프레임워크 <br/>- PyTorch: 멀티모달 모델 로딩 및 추론 <br/>- CUDA/cuDNN: 가속 연산 및 커널 코드 구현 <br/><br/>2. 커널 최적화 및 경량화 툴 <br/>- TensorRT: 모델 경량화 및 inference graph 최적화 <br/>- ONNX / ONNX Runtime: 모델 변환 및 다양한 하드웨어에서 추론 가능 <br/><br/>3. Profiling & 분석 도구 <br/>- Nsight Systems: GPU/CPU layer 별 latency 및 kernel-level utilization 분석 <br/>- Nsight Compute: Kernel level utilization 분석 <br/>- PyTorch Profiler: 연산별 latency와 memory 사용량 분석 <br/>- TorchMetrics: multimodal 성능 평가 |
| 사용할 소프트웨어 패키지의 명칭과 URL | - PyTorch: https://pytorch.org <br/>- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit <br/>- TensorRT: https://developer.nvidia.com/tensorrt <br/>- ONNX Runtime: https://onnxruntime.ai <br/>- Nsight Systems: https://developer.nvidia.com/nsight-systems <br/>- Nsight Compute: https://developer.nvidia.com/nsight-compute  <br/>- TorchMetrics: https://lightning.ai/docs/torchmetrics/stable/ |
| 팀그라운드룰 URL | https://github.com/Nitro-Ewha/Multimodal_on_Jetson/blob/main/GroudRule.MD |
| 최종수정일 | 2025.12.15. |
