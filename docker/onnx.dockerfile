# CUDA 기반 이미지를 Ubuntu 22.04로 변경
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 환경변수 설정
ENV DEBIAN_FRONTEND=noninteractive

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip3 install --no-cache-dir \
    onnxruntime-gpu \
    numpy \
    pillow

# 작업 디렉토리 설정
WORKDIR /app

# 컨테이너 실행 시 기본 명령어
CMD ["bash"]