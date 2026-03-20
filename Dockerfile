FROM nvcr.io/nvidia/tensorrt-llm/release:1.2.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            curl \
            ca-certificates \
            dropbear \
        && \
    rm -rf /var/lib/apt/lists/* /etc/dropbear && \
    useradd -m -s /bin/bash fox

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
