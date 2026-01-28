# Use NVIDIA PyTorch image that already has CUDA 12.8 + cuDNN 9 + Blackwell support
# FROM nvcr.io/nvidia/pytorch:25.01-py3

# WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     ffmpeg \
#     libsm6 \
#     libxext6 \
#     libgl1 \
#     curl \
#     git \
#     unzip \
#     libglib2.0-0 \
#     libxrender-dev \
#     libgomp1 \
#  && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .

# RUN pip install --upgrade pip --break-system-packages && \
#     pip install --break-system-packages "numpy<2.0" scikit-image albumentations omegaconf && \
#     pip install --break-system-packages --no-cache-dir -r requirements.txt && \
#     pip uninstall -y onnxruntime || true && \
#     pip uninstall -y onnxruntime-gpu || true && \
#     pip install --break-system-packages --no-deps onnxruntime-gpu==1.23.2 insightface opencv-python && \
#     pip install --break-system-packages numpy==1.26.4 && \
#     pip uninstall -y onnxruntime || true

# COPY . .

# CMD ["python", "-m", "src.api.app"]





FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Install Python dependencies - PyTorch with CUDA first, then requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git


# Copy project files
COPY . .

# Default command
CMD ["python", "-m", "src.api.app"]




# # 09/12/25
# FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

# WORKDIR /app

# # Add deadsnakes PPA for Python 3.12
# RUN apt-get update && apt-get install -y \
#     software-properties-common \
#     && add-apt-repository ppa:deadsnakes/ppa \
#     && apt-get update

# # Install Python 3.12 and dependencies
# RUN apt-get install -y \
#     python3.12 \
#     python3.12-venv \
#     python3.12-dev \
#     python3-pip \
#     build-essential \
#     ffmpeg \
#     libsm6 \
#     libxext6 \
#     libgl1 \
#     curl \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Set Python 3.12 as default
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
#     update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# # Upgrade pip for Python 3.12
# RUN python3.12 -m ensurepip && \
#     python3.12 -m pip install --upgrade pip

# # Copy the custom ONNX Runtime wheel
# COPY onnx-custom/onnxruntime_gpu-1.24.0-cp312-cp312-linux_x86_64.whl /tmp/

# # Install PyTorch with CUDA 12.1 (closest available to CUDA 13.0)
# RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# # Install the custom ONNX Runtime wheel
# RUN pip install --no-cache-dir /tmp/onnxruntime_gpu-1.24.0-cp312-cp312-linux_x86_64.whl

# # Copy requirements.txt and install remaining dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Clean up
# RUN rm /tmp/onnxruntime_gpu-1.24.0-cp312-cp312-linux_x86_64.whl

# # Copy project files
# COPY . .

# CMD ["python", "-m", "src.api.app"]


# FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

# WORKDIR /app
# ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# # 1. System dependencies + Python 3.12
# RUN apt-get update && apt-get install -y \
#     tzdata \
#     software-properties-common \
#     build-essential \
#     ffmpeg \
#     libsm6 \
#     libxext6 \
#     libgl1 \
#     curl \
#     git \
#     wget && \
#     add-apt-repository ppa:deadsnakes/ppa -y && \
#     apt-get update && apt-get install -y \
#     python3.12 \
#     python3.12-dev \
#     python3.12-venv \
#     && rm -rf /var/lib/apt/lists/*

# # 2. Set Python 3.12 as default
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
#     update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# # 3. Install pip for Python 3.12
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# # 4. Upgrade pip tools
# RUN python3.12 -m pip install --no-cache-dir --upgrade pip setuptools setuptools-scm wheel packaging

# # 5. Install PyTorch with CUDA 12.1
# RUN python3.12 -m pip install --no-cache-dir \
#     torch torchvision torchaudio \
#     --index-url https://download.pytorch.org/whl/cu121

# # 5.5 NUCLEAR CLEAN INSTALL - Wipe ALL onnxruntime + cache FIRST
# COPY onnx-custom/onnxruntime_gpu-1.24.0-cp312-cp312-linux_x86_64.whl /tmp/
# RUN python3.12 -m pip uninstall -y onnxruntime onnxruntime-gpu insightface || true && \
#     python3.12 -m pip cache purge && \
#     python3.12 -m pip install --no-cache-dir --force-reinstall /tmp/onnxruntime_gpu-1.24.0-cp312-cp312-linux_x86_64.whl && \
#     rm -f /tmp/onnxruntime_gpu-1.24.0-cp312-cp312-linux_x86_64.whl && \
#     python3.12 -c "import onnxruntime as ort; print('✅ CUSTOM ONNX:', ort.__version__, ort.__file__)" || echo "❌ CUSTOM ONNX FAILED"

# # 6. Install requirements WITHOUT onnxruntime/insightface deps using --no-deps
# COPY requirements.txt .
# RUN python3.12 -m pip install --no-cache-dir -r requirements.txt --no-deps

# # 6.5 Install insightface LAST with NO dependencies (uses custom ONNX)
# RUN python3.12 -m pip install --no-cache-dir --no-deps insightface && \
#     python3.12 -c "import insightface; import onnxruntime; print('✅ INSIGHTFACE + ONNX WORKING')" || echo "❌ INSIGHTFACE FAILED"

# # 7. Copy project source
# COPY . .

# # 8. Run the API
# CMD ["python", "-m", "src.api.app"]


