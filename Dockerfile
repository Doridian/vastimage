FROM nvcr.io/nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends libpython3-dev python3 python3-pip dropbear curl && \
    rm -rf /var/lib/apt/lists/* /etc/dropbear && \
    useradd -m -s /bin/bash fox

RUN pip3 install --break-system-packages --ignore-installed pip setuptools wheel && pip3 install --break-system-packages tensorrt_llm huggingface-hub

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
