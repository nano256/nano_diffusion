#!/bin/bash

# Detect OS
OS=$(uname -s)
if [ "$OS" = "Linux" ]; then
    GPU_FLAG="--gpus all"
else
    GPU_FLAG=""
fi

# Function to display help
show_help() {
    echo "Usage: ./docker-run.sh [OPTION]"
    echo "Options:"
    echo "  build-dev    Build the development container"
    echo "  build-train  Build the training container"
    echo "  run-dev      Run the development container"
    echo "  run-train    Run the training container"
    echo "  jupyter      Run jupyter notebook in the training container"
    echo "  clean        Clean up containers and images"
    echo "  reset-dev    Reset devcontainer environment (fix VSCode issues)"
    echo "  help         Display this help message"
}

# Check if command is provided
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# Process command
case "$1" in
    build-dev)
        echo "Building development container..."
        docker build -t nano-diffusion-dev -f .devcontainer/Dockerfile .
        ;;
    build-train)
        echo "Building training container..."
        docker build -t nano-diffusion-train -f Dockerfile .
        ;;
    run-dev)
        echo "Running development container..."
        docker run $GPU_FLAG -v $(pwd):/workspace -it nano-diffusion-dev bash
        ;;
    run-train)
        echo "Running training container..."
        docker run $GPU_FLAG -v $(pwd):/workspace -v $(pwd)/data:/workspace/data -p 6006:6006 -p 8888:8888 -it nano-diffusion-train bash
        ;;
    jupyter)
        echo "Running jupyter notebook in training container..."
        docker run $GPU_FLAG -v $(pwd):/workspace -v $(pwd)/data:/workspace/data -p 8888:8888 -it nano-diffusion-train jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
        ;;
    clean)
        echo "Cleaning up containers and images..."
        docker rm $(docker ps -a -q --filter ancestor=nano-diffusion-dev --filter ancestor=nano-diffusion-train) 2>/dev/null || true
        docker rmi nano-diffusion-dev nano-diffusion-train 2>/dev/null || true
        ;;
    reset-dev)
        echo "Resetting devcontainer environment..."
        # Remove any containers related to this project
        docker ps -a -q --filter label=devcontainer.local_folder=$(pwd) | xargs docker rm -f 2>/dev/null || true
        # Remove any devcontainer images
        docker images -q | xargs docker rmi -f 2>/dev/null || true
        echo "Devcontainer environment reset. You can now reopen in container."
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac 