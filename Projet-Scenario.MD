### Team 29 Nitro

| 항목 | 내용 |
| --- | --- |
| 프로젝트명 | **Profiling Edge-Optimized Multimodal Model Architecture for Accelerated Inference on Jetson Nano** |
| 프로젝트 키워드 | Edge-Optimized Multimodal model, Inference Acceleration, Edge Device, Embedded AI, Profiling |
| 트랙 | 연구 |
| 프로젝트 멤버 | 박세진, 박윤형, 강선화 |
| 팀 지도교수 | 윤명국 |
| 무엇을 만들고자 하는가 | **엣지 디바이스(Jetson Nano)**에서 실행 가능한 **경량화 멀티모달 모델**을 대상으로, **하드웨어 병목을 분석하고 아키텍처 수준에서 최적화**함으로써 **추론(inference) 속도를 향상**시키는 시스템을 구축하는 것이다. |
| 고객 | Jetson Nano와 같이 컴퓨팅 자원이 제한된 소형 임베디드 시스템에서, 계산 복잡도가 높은 멀티모달(Multimodal) task를 수행하고자 하는 연구원 및 개발자 |
| Pain Point | 최근 Vision-Language(VL) 모델의 급격한 발전에도 불구하고, 엣지 디바이스 환경에서 멀티모달 입력(RGB, depth, IR 등)을 효율적으로 처리하고, 제한된 연산 자원과 메모리 제약을 극복하는 것은 여전히 큰 도전이다.
기존 연구들은 Knowledge Distillation, Model Pruning, Quantization 등 소프트웨어 기반 경량화 기법을 통해 모델 크기와 연산량을 줄였으나, 경량화된 모델을 실제 엣지 디바이스에 배치했을 때, 레이어별 HW 병목(layer-level latency bottleneck)을 분석하고 아키텍처 수준에서 최적화하여 inference 속도를 극대화하는 연구는 미흡하다.
엣지 친화적 멀티모달 모델 구현에서는 단순한 소프트웨어 경량화뿐 아니라, **디바이스 하드웨어 특성을 고려한 병목 분석 및 아키텍처-레벨 최적화가 핵심적인 연구 과제**로 남아 있다. |
| 사용할 소프트웨어 패키지의 명칭과 핵심기능/용도 | **1. 멀티모달 모델 프레임워크**
- PyTorch: 멀티모달 모델 로딩 및 추론
- CUDA/cuDNN: 가속 연산 지원
- EdgeVL: Jetson Nano용 경량화 VLM, 멀티 모달 입력 지원, baseline inference

**2. 엣지 디바이스 최적화**
- NVIDIA TensorRT: GPU/Jetson용 inference 최적화, 연산 병렬화, FP16/INT8 양자화 지원
- ONNX / ONNX Runtime: 모델 변환 및 다양한 하드웨어에서 추론, 연산 최적화

**3. 프로파일러**
- Nsight Systems: Jetson Nano GPU/CPU layer별 latency 분석, 병목 지점 시각화
- PyTorch Profiler: 연산별 latency, memory 사용량 분석, layer별 bottleneck 확인
- TorchMetrics: 모델 성능 평가, accuracy/CLIP-score 등 task-specific metric 계산 |
| 사용할 소프트웨어 패키지의 명칭과 URL | - PyTorch: https://pytorch.org/
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- EdgeVL: https://github.com/ramdrop/edgevl
- NVIDIA TensorRT: https://developer.nvidia.com/tensorrt
- ONNX / ONNX Runtime: [https://onnxruntime.ai](https://onnxruntime.ai/)
- NVIDIA NSight: https://developer.nvidia.com/nsight-systems
- PyTorch Profiler: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- TorchMetrics: https://lightning.ai/docs/torchmetrics/stable/ |
| 팀그라운드룰 URL | https://github.com/Nitro-Ewha/Multimodal_on_Jetson/blob/main/GroudRule.MD |
| 최종수정일 | 2025.09.16. |
