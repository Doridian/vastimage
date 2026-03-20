#!/bin/sh
set -e

. /run/container-env

exec trtllm-serve "${MODEL_NAME}" \
    --host 127.0.0.1 \
    --port 8080 \
    "$@"
