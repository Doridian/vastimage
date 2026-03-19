FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN mkdir -p /models
COPY entrypoint.sh /entrypoint.sh
ENV LLAMA_MODEL_URL=https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-UD-Q4_K_XL.gguf

ENTRYPOINT [ "/entrypoint.sh" ]
CMD []
