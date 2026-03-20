FROM nvcr.io/nvidia/tensorrt-llm/release:1.2.0


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            curl \
            ca-certificates \
            dropbear \
            libnvidia-ml-dev \
        && \
    rm -rf /var/lib/apt/lists/* /etc/dropbear && \
    useradd -m -s /bin/bash fox

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH

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
