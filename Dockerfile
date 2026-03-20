FROM nvcr.io/nvidia/cuda:12.6.3-cudnn9-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip dropbear curl && \
    rm -rf /var/lib/apt/lists/* /etc/dropbear && \
    useradd -m -s /bin/bash fox

RUN pip3 install tensorrt-llm huggingface-hub

COPY rootfs/ /

RUN chown -R fox:fox /home/fox && \
    chmod 700 /home/fox /home/fox/.ssh && \
    chmod 600 /home/fox/.ssh/authorized_keys

ENV MODEL_NAME=Qwen/Qwen2.5-Coder-32B-Instruct
ENV HF_HOME=/models

EXPOSE 2222/tcp
VOLUME /models
VOLUME /etc/dropbear

ENTRYPOINT [ "/entrypoint.sh" ]
CMD []
