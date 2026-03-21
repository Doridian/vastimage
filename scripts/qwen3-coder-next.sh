#!/bin/sh
set -ex

. /run/container-env
export HOME=/models

ulimit -s unlimited
#ulimit -l unlimited
ulimit -a

cd /app
exec ./llama-server --host 127.0.0.1 --port 8080 -hf unsloth/Qwen3-Coder-Next-GGUF:UD-Q8_K_XL --threads "$(nproc)" "$@"
