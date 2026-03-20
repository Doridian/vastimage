#!/usr/bin/env python3
"""
vast_instance.py

Starts a Vast.ai instance and opens an SSH tunnel, then waits until interrupted.

Behavior:
- Uses a fixed GPU type per quantization mode (configured in PICK_CONFIGS)
- Reuse an existing instance of the selected GPU type if one exists
- Otherwise create a new instance only if price <= MAX_HOURLY_PRICE
- Start the instance and wait for SSH port to become available
- Open an SSH local-port-forward tunnel (localhost:6969 -> remote:8080)
- Stop or destroy the instance on exit

Requirements:
    pip install httpx

Example:
    export VAST_API_KEY="..."
    export MODEL_NAME="Qwen/Qwen3-Coder-Next"
    export MAX_HOURLY_PRICE="1.00"
    export INSTANCE_ACTION="stop"   # or "destroy"

    python main.py --pick int4-awq
    python main.py --pick fp8
"""

import argparse
import asyncio
import os

import httpx

import instance as instance_module
from config import INSTANCE_ACTION, PICK_CONFIGS, log
from vast_api import vast_destroy_instance, vast_manage_instance


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start a Vast.ai instance and serve a HuggingFace model with TensorRT-LLM.")
    parser.add_argument(
        "--pick", required=True, choices=list(PICK_CONFIGS),
        help="Quantization mode to use.")
    args = parser.parse_args()

    await instance_module.ensure_instance_ready(args.pick)
    log.info("Instance ready. Press Ctrl+C to stop.")
    try:
        if instance_module.ssh_process is not None:
            await instance_module.ssh_process.wait()
        else:
            await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        if instance_module.ssh_process is not None:
            log.info("Terminating SSH tunnel")
            instance_module.ssh_process.terminate()
            try:
                await asyncio.wait_for(instance_module.ssh_process.wait(), timeout=5.0)
            except Exception:
                instance_module.ssh_process.kill()
        if instance_module.known_hosts_file is not None:
            os.unlink(instance_module.known_hosts_file)
        if instance_module.current_instance_id is not None:
            async with httpx.AsyncClient(timeout=60.0) as client:
                if INSTANCE_ACTION == "destroy":
                    log.info("Destroying instance %s", instance_module.current_instance_id)
                    await vast_destroy_instance(client, instance_module.current_instance_id)
                else:
                    log.info("Stopping instance %s", instance_module.current_instance_id)
                    await vast_manage_instance(client, instance_module.current_instance_id, "stopped")


if __name__ == "__main__":
    asyncio.run(main())
