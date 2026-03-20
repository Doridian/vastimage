ARG CUDA_MAJOR=13
ARG CUDA_MINOR=0.0
FROM nvcr.io/nvidia/cuda:${CUDA_MAJOR}.${CUDA_MINOR}-cudnn-runtime-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            ca-certificates \
            curl \
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

RUN pip3 install --break-system-packages --ignore-installed pip setuptools wheel && \
    pip3 install --break-system-packages tensorrt_llm huggingface-hub

COPY rootfs/ /

RUN chown -R fox:fox /home/fox && \
    chmod 700 /home/fox /home/fox/.ssh && \
    chmod 600 /home/fox/.ssh/authorized_keys

ENV HF_HOME=/models
ENV OPAL_PREFIX=

EXPOSE 2222/tcp
VOLUME /models
VOLUME /etc/dropbear

ENTRYPOINT [ "/entrypoint.sh" ]
CMD []
