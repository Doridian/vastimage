FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt-get update && \
    apt-get install -y --no-install-recommends dropbear && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -s /bin/bash fox

COPY rootfs/ /

RUN chown -R fox:fox /home/fox && \
    chmod 700 /home/fox /home/fox/.ssh && \
    chmod 600 /home/fox/.ssh/authorized_keys

ENV LLAMA_MODEL_URL=https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-UD-Q4_K_XL.gguf

EXPOSE 2222/tcp
VOLUME /models
VOLUME /etc/dropbear

ENTRYPOINT [ "/entrypoint.sh" ]
CMD []
