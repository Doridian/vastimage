#!/bin/sh
set -e

. /run/container-env

if [ ! -f /models/download.gguf ]; then
    echo "Downloading model from ${LLAMA_MODEL_URL}"
    curl -fL -o /models/download.gguf "${LLAMA_MODEL_URL}"
else
    echo 'Model already exists at /models/download.gguf, skipping download.'
fi

exec /app/llama-server --host 127.0.0.1 --port 8080 --model /models/download.gguf --threads "$(nproc)" "$@"
