FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    sudo \
    gnupg \
    lsb-release \
    ca-certificates \
    zsh \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Set zsh as default shell
RUN chsh -s /bin/zsh root

# Install Python development tools
RUN pip3 install --no-cache-dir \
    black \
    flake8 \
    pylint \
    pytest \
    ipython \
    jupyter

# Set working directory
WORKDIR /workspace

# Set default shell to zsh
SHELL ["/bin/zsh", "-c"]

# Keep the container running
CMD ["zsh"] 