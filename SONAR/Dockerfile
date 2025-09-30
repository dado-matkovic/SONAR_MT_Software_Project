# Base image with PyTorch and CUDA pre-installed
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda

# Install additional programs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    htop \
    gnupg \
    curl \
    ca-certificates \
    vim \
    tmux && \
    rm -rf /var/lib/apt/lists/*

# Update pip
RUN python3 -m pip install --upgrade pip

# Install Python dependencies from requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# Set environment variable for Python locale
ENV LANG=C.UTF-8

# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

# Create the user
RUN useradd -l -m -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME
# this will fix a wandb issue
RUN mkdir /home/$USER_NAME/.local 

# Change owner of home dir (Note: this is not the lsv nethome)
RUN chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

# Set cache environment variables
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets \
    WANDB_DIR=/app/wandb \
    WANDB_CACHE_DIR=/app/wandb/.cache \
    WANDB_CONFIG_DIR=/app/wandb/.config \
    TRITON_CACHE_DIR=/tmp/triton \
    TORCH_HOME=/app/.cache/torch \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create all needed directories with correct ownership and permissions
RUN mkdir -p /app/wandb \
             /app/sonar_checkpoints \
             /app/outputs \
             /app/.cache/huggingface \
             /tmp/triton \
             /home/$USER_NAME/.local && \
    chown -R ${USER_UID}:${USER_GID} /app /home/$USER_NAME && \
    chmod -R 777 /app
# Set working directory
WORKDIR /app

# Set the default working directory
COPY . /app
COPY config/ /app/config/
COPY nllb_datasets/ /app/nllb_datasets/

# Default command to run the application
CMD ["/bin/bash", "/app/run.sh"]

#ARG WANDB_API_KEY
#ENV WANDB_API_KEY=${WANDB_API_KEY}