#!/bin/sh
set -e

exec llama-server --host :: --port "${LLAMA_PORT:-8080}" --model "/models/${LLAMA_MODEL}.gguf" --threads "$(nproc)"
