FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /workspace

# Copy requirements from setup.py
COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Install development tools
RUN pip install --no-cache-dir \
    black \
    flake8 \
    pylint \
    pytest \
    ipython \
    jupyter \
    tensorboard

# Copy the rest of the code
COPY . .

# Default command
CMD ["bash"] 