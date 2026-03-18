FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN mkdir -p /models
ADD https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-UD-Q4_K_XL.gguf /models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf

COPY entrypoint.sh /entrypoint.sh
ENV LLAMA_MODEL=Qwen3-Coder-Next-UD-Q4_K_XL

ENTRYPOINT [ "/entrypoint.sh" ]
