.PHONY: build-dev build-train run-dev run-train clean

# Detect OS
UNAME_S := $(shell uname -s)

# Set GPU flags based on OS
ifeq ($(UNAME_S),Linux)
	GPU_FLAG := --gpus all
else
	GPU_FLAG :=
endif

# Build the development container
build-dev:
	docker build -t nano-diffusion-dev -f .devcontainer/Dockerfile .

# Build the training container
build-train:
	docker build -t nano-diffusion-train -f Dockerfile .

# Run the development container
run-dev:
	docker run $(GPU_FLAG) -v $(PWD):/workspace -it nano-diffusion-dev bash

# Run the training container
run-train:
	docker run $(GPU_FLAG) -v $(PWD):/workspace -v $(PWD)/data:/workspace/data -p 6006:6006 -p 8888:8888 -it nano-diffusion-train bash

# Run jupyter notebook in the training container
run-jupyter:
	docker run $(GPU_FLAG) -v $(PWD):/workspace -v $(PWD)/data:/workspace/data -p 8888:8888 -it nano-diffusion-train jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

# Clean up containers and images
clean:
	docker rm $$(docker ps -a -q --filter ancestor=nano-diffusion-dev --filter ancestor=nano-diffusion-train) 2>/dev/null || true
	docker rmi nano-diffusion-dev nano-diffusion-train 2>/dev/null || true 