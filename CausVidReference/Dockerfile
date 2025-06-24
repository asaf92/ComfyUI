FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04


RUN apt-get update && apt-get install -y \
    python3 python3-pip git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision torchaudio --no-cache-dir \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Clone and build FlashAttention from source
# This can take some time and is CPU-intensive
ENV FLASH_ATTN_CUDA_ARCHS="86;120" 
ENV TORCH_CUDA_ARCH_LIST="8.6+PTX;12.0+PTX" 
ENV MAX_JOBS=10
RUN pip3 install --no-cache-dir packaging ninja && \
    pip3 install --no-build-isolation --no-cache-dir flash-attn && \
    rm -rf /tmp/*

WORKDIR /workspace/CausVid
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
