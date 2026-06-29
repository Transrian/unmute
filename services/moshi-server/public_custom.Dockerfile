# Compile moshi-server at build time so startup is near-instant.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS base

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies, including dos2unix to handle Windows line endings
# Use deadsnakes PPA to get Python 3.12 so pyo3 (Rust↔Python) links against
# the same version uv uses for the virtual environment.
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    curl \
    build-essential \
    ca-certificates \
    libssl-dev \
    libpython3.12-dev \
    python3.12 \
    python3.12-venv \
    git \
    pkg-config \
    cmake \
    wget \
    openssh-client \
    dos2unix \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Tell pyo3-build-config to use Python 3.12 so the Rust binary links against
# the same Python version that uv will use for the virtual environment.
ENV PYTHON_SYS_EXECUTABLE=/usr/bin/python3.12
ENV PYO3_PYTHON=/usr/bin/python3.12

COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx /bin/

WORKDIR /app

# Download pyproject.toml and uv.lock for the Python environment
RUN wget https://raw.githubusercontent.com/kyutai-labs/moshi/4fae088e130f6b44d489aefc0ef1836745e921de/rust/moshi-server/pyproject.toml
RUN wget https://raw.githubusercontent.com/kyutai-labs/moshi/4fae088e130f6b44d489aefc0ef1836745e921de/rust/moshi-server/uv.lock

# Compile moshi-server with CUDA support at build time.
# CUDA_COMPUTE_CAP tells candle-kernels which GPU arch to compile for,
# bypassing the nvidia-smi call that fails during docker build (no GPU driver).
# Common values: 75=Turing, 80=Ampere, 86=Ada, 90=Hopper, 120=Blackwell
ARG CUDA_COMPUTE_CAP=86
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}

RUN cargo install --features cuda --locked moshi-server@0.6.4

# Bake the Python virtual environment into the image so uv run isn't needed at startup.
# This installs huggingface_hub and other deps the Rust binary needs at runtime.
RUN uv venv --python python3.12 && uv sync --locked
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . .

# Ensure the startup script is runnable inside the container.
# This prevents script errors that can happen if the project was cloned on Windows,
# which uses a different text file format (CRLF) than the Linux environment in the container (LF).
RUN dos2unix ./start_moshi_server_public_custom.sh && chmod +x ./start_moshi_server_public_custom.sh

HEALTHCHECK --start-period=15s \
    CMD curl --fail http://localhost:8080/api/build_info || exit 1

EXPOSE 8080
ENV RUST_BACKTRACE=1

ENTRYPOINT ["./start_moshi_server_public_custom.sh"]
