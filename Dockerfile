FROM ubuntu:24.04

ARG CUDA_MAJOR=13
ARG CUDA_MINOR=0
ENV CUDA_MAJOR=${CUDA_MAJOR}
ENV CUDA_MINOR=${CUDA_MINOR}

RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
            curl \
            ca-certificates \
        && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    rm /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
            cuda-compat-${CUDA_MAJOR}-${CUDA_MINOR} \
            cuda-libraries-${CUDA_MAJOR}-${CUDA_MINOR} \
            dropbear \
            libcudnn9-cuda-${CUDA_MAJOR} \
            libnvidia-ml-dev \
            libopenmpi-dev \
            libpython3-dev \
            openmpi-bin \
            openmpi-common \
            python3 \
            python3-pip \
        && \
    rm -rf /var/lib/apt/lists/* /etc/dropbear && \
    useradd -m -s /bin/bash fox

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:$LD_LIBRARY_PATH

RUN pip3 install --break-system-packages --ignore-installed pip setuptools wheel && \
    pip3 install --break-system-packages tensorrt_llm huggingface-hub

COPY rootfs/ /

RUN chown -R fox:fox /home/fox && \
    chmod 700 /home/fox /home/fox/.ssh && \
    chmod 600 /home/fox/.ssh/authorized_keys

ENV HF_HOME=/models

EXPOSE 2222/tcp
VOLUME /models
VOLUME /etc/dropbear

ENTRYPOINT [ "/entrypoint.sh" ]
CMD []
