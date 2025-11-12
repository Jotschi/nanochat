FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ncdu \
    htop \
    nvtop \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/build
WORKDIR /opt/build