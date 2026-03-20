#!/bin/sh
set -e

# w4a16_awq, fp8

. /run/container-env

exec mpirun trtllm-serve 'Qwen/Qwen3-Coder-Next' \
    --host 127.0.0.1 \
    --port 8080 \
    w4a16_awq
