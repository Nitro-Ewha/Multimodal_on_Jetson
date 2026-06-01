### Team 29 Nitro

| 항목 | 내용 |
| --- | --- |
| 프로젝트명 | **Accelerating Vision Transformer Inference via Non-GEMM Kernel Fusion on Edge GPUs** |
| 프로젝트 키워드 | NVIDIA Jetson, Multimodal, CUTLASS, Kernel Fusion, Embedded GPU Optimization |
| 트랙 | 연구 |
| 프로젝트 멤버 | 박세진, 박윤형, 강선화 |
| 팀 지도교수 | 윤명국 교수님 |
| 프로젝트 소개 | 엣지 GPU 환경에서 Vision Transformer 기반 멀티모달 모델을 추론할 때 발생하는 GPU kernel-level 병목을 분석하고, 비-GEMM 연산을 GEMM 인접 실행 경로에 흡수하는 Kernel Fusion 기법을 통해 추론 지연시간을 줄이는 시스템 최적화 연구이다. |
| 타겟 사용자 | 엣지 GPU 환경에서 멀티모달 Transformer 모델의 inference 성능 최적화에 관심 있는 연구자, GPU 기반 AI 시스템 개발자, Vision-Language Model을 제한된 하드웨어 자원에서 실시간으로 실행하고자 하는 임베디드 AI 개발자 |
| 문제 정의 | 최근 멀티모달 모델은 Vision Transformer 기반 구조를 활용하며, 모델 규모와 visual token 수 증가로 인해 연산량과 메모리 요구량이 크게 증가하고 있다. 그러나 엣지 GPU는 서버급 GPU에 비해 연산 자원과 메모리 대역폭이 제한적이어서, Transformer 추론 시 kernel launch overhead와 DRAM access가 latency 증가의 주요 원인이 된다. 특히 QKV Split&Transpose, LayerNorm, Bias/Residual Add와 같은 비-GEMM 연산은 계산량은 작지만 별도 CUDA kernel로 실행되며 반복적인 메모리 접근을 유발한다. TensorRT와 CUTLASS는 일부 연산 패턴에 대해 최적화를 제공하지만, 이러한 비-GEMM 커널 병목을 모두 제거하기에는 한계가 있다. 따라서 본 연구는 엣지 GPU 환경에서 발생하는 비-GEMM 커널의 launch overhead와 DRAM access를 주요 병목으로 정의하고, 이를 Kernel Fusion으로 줄이고자 한다. |
| Solution | 본 연구는 CUTLASS를 활용하여 Transformer 블록 내 연산들을 하나로 통합하는 커널 퓨전(Kernel Fusion) 솔루션을 제안한다.<br/>첫째로, QKV 프로젝션 GEMM의 출력 단계에서 데이터 분할 및 전치 연산을 동시에 처리하여 메모리 병목을 제거한다. 둘째로, Bias/Residual Add와 LayerNorm 연산을 앞뒤 GEMM의 Epilogue 및 Mainloop 단계에 분산 통합함으로써, 거대한 중간 활성화 텐서의 DRAM 전역 메모리 접근을 최소화한다.<br/>이를 통해 특히 자원이 제한된 엣지 디바이스 환경에서 멀티모달 모델의 추론 속도를 효율적으로 개선한다. |
| 팀그라운드룰 URL | https://github.com/Nitro-Ewha/Multimodal_on_Jetson/blob/main/GroudRule.MD |
| 최종수정일 | 2026.06.01. |
