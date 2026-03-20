#!/bin/sh
set -e

. /run/container-env

if [ -n "${TRTLLM_QUANT:-}" ]; then
    set -- --quantization "${TRTLLM_QUANT}" "$@"
fi

exec trtllm-serve "${MODEL_NAME}" \
    --host 127.0.0.1 \
    --port 8080 \
    "$@"
