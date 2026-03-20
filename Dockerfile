ARG CUDA_MAJOR=13
ARG CUDA_MINOR=0.0
FROM nvcr.io/nvidia/cuda:${CUDA_MAJOR}.${CUDA_MINOR}-cudnn-runtime-ubuntu24.04

ENV CUDA_MAJOR=${CUDA_MAJOR}
ENV CUDA_MINOR=${CUDA_MINOR}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            curl \
            ca-certificates \
            dropbear \
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
