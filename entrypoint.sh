#!/bin/sh
set -e

curl -fL -o /models/download.gguf "${LLAMA_MODEL_URL}"

exec llama-server --host :: --port "${LLAMA_PORT:-8080}" --model /models/download.gguf --threads "$(nproc)"
