#!/usr/bin/env python3
"""
vast_instance.py

Starts a Vast.ai instance and opens an SSH tunnel, then waits until interrupted.

Behavior:
- Reuse an existing instance with the matching label and GPU type if one exists
- Otherwise create a new instance only if price <= --max-hourly-price
- Start the instance and wait for SSH port to become available
- Open an SSH local-port-forward tunnel (localhost:LOCAL_PORT -> remote:8080)
- Stop or destroy the instance on exit

Requirements:
    pip install httpx

Example:
    export VAST_API_KEY="..."

    python main.py --gpu-search H100 --model Qwen/Qwen3-Coder-Next
    python main.py --gpu-search H200 --gpu-exclude --model Qwen/Qwen3-Coder-Next --max-hourly-price 2.00
"""

import argparse
import asyncio
import os

import config
from instance import Instance
from vast_api import VastAPI


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start a Vast.ai instance and serve a model via SSH tunnel.")

    parser.add_argument("--gpu-search", required=True,
                        help="GPU name to search for (e.g. H100, H200).")
    parser.add_argument("--gpu-exclude", nargs="*", default=["PCIE"],
                        help="GPU name tokens to exclude (e.g. PCIE). (default: PCIE)")

    parser.add_argument("--instance-action", default="stop", choices=["stop", "destroy"],
                        help="Action on exit: stop or destroy the instance. (default: stop)")
    parser.add_argument("--instance-label", default="vastimage-controlled",
                        help="Label for the instance, used to identify reusable instances. (default: vastimage-controlled)")
    parser.add_argument("--vast-api-base", default="https://console.vast.ai/api/v0",
                        help="Vast.ai API base URL. (default: https://console.vast.ai/api/v0)")
    parser.add_argument("--startup-timeout", type=int, default=3600,
                        help="Seconds to wait for instance to become ready. (default: 3600)")
    parser.add_argument("--healthcheck-interval", type=float, default=3.0,
                        help="Seconds between status polls. (default: 3.0)")
    parser.add_argument("--max-hourly-price", type=float, default=2.00,
                        help="Maximum accepted price in $/hr. (default: 2.00)")
    parser.add_argument("--prefer-verified", default=True, action=argparse.BooleanOptionalAction,
                        help="Prefer verified Vast.ai hosts. (default: true)")
    parser.add_argument("--require-reliability-gte", type=float, default=0.95,
                        help="Minimum host reliability score [0, 1]. (default: 0.95)")
    parser.add_argument("--search-limit", type=int, default=500,
                        help="Maximum number of offers to fetch. (default: 500)")
    parser.add_argument("--vast-image", default="ghcr.io/doridian/vastimage/vastimage:latest",
                        help="Docker image to run on the instance.")
    parser.add_argument("--vast-disk-gb", type=float, default=100.0,
                        help="Disk size in GB for the instance. (default: 100)")
    parser.add_argument("--local-port", type=int, default=6969,
                        help="Local port for the SSH tunnel. (default: 6969)")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level. (default: INFO)")
    parser.add_argument("--script", default="qwen3-coder-next.sh",
                        help="Server startup script template. (default: qwen3-coder-next.sh)")

    args = parser.parse_args()
    
    if not config.VAST_API_KEY:
        raise RuntimeError("VAST_API_KEY environment variable is required")

    with open(os.path.join("scripts", args.script), "r") as f:
        script_template = f.read()

    api = VastAPI(config.VAST_API_KEY, api_base=args.vast_api_base)
    instance = Instance(
        api,
        model_name=args.model,
        gpu_search=args.gpu_search,
        gpu_exclude=args.gpu_exclude,
        script_template=script_template,
        max_hourly_price=args.max_hourly_price,
        startup_timeout=args.startup_timeout,
        healthcheck_interval=args.healthcheck_interval,
        instance_action=args.instance_action,
        vast_image=args.vast_image,
        vast_disk_gb=args.vast_disk_gb,
        instance_label=args.instance_label,
        local_port=args.local_port,
        prefer_verified=args.prefer_verified,
        require_reliability_gte=args.require_reliability_gte,
        search_limit=args.search_limit,
    )

    async with instance.start():
        config.log.info("Instance ready. Press Ctrl+C to stop.")
        try:
            await instance.wait()
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
