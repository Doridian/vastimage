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
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

# -----------------------------------------------------------------------------
# Remote server script (sent over SSH stdin on connect)
# -----------------------------------------------------------------------------

SERVER_SH = """\
#!/bin/sh
set -e

exec trtllm-serve "{model_name}" \\
    --host 127.0.0.1 \\
    --port 8080 \\
    {quant_flag}
"""

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

VAST_API_KEY = os.environ.get("VAST_API_KEY", "").strip()
VAST_API_BASE = os.environ.get("VAST_API_BASE", "https://console.vast.ai/api/v0").rstrip("/")

LOCAL_LLAMA_PORT = 6969

STARTUP_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT", "900"))
HEALTHCHECK_INTERVAL = float(os.environ.get("HEALTHCHECK_INTERVAL", "3"))

# Instance selection / creation
MAX_HOURLY_PRICE = float(os.environ.get("MAX_HOURLY_PRICE", "1.00"))
PREFER_VERIFIED = os.environ.get("PREFER_VERIFIED", "true").lower() == "true"
REQUIRE_RELIABILITY_GTE = float(os.environ.get("REQUIRE_RELIABILITY_GTE", "0.95"))
SEARCH_LIMIT = int(os.environ.get("SEARCH_LIMIT", "500"))

VAST_IMAGE = os.environ.get("VAST_IMAGE", "ghcr.io/doridian/vastimage/vastimage:latest").strip()

# Create-instance config
VAST_DISK_GB = float(os.environ.get("VAST_DISK_GB", "100"))
INSTANCE_LABEL = os.environ.get("INSTANCE_LABEL", "vastimage-controlled").strip()

# TensorRT-LLM model config
MODEL_NAME = os.environ.get("MODEL_NAME", "").strip()

# Shutdown action
INSTANCE_ACTION = os.environ.get("INSTANCE_ACTION", "stop").strip().lower()  # stop | destroy

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

if not VAST_API_KEY:
    raise RuntimeError("VAST_API_KEY is required")
if not MODEL_NAME:
    raise RuntimeError("MODEL_NAME is required")
if INSTANCE_ACTION not in {"stop", "destroy"}:
    raise RuntimeError("INSTANCE_ACTION must be 'stop' or 'destroy'")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("vast-instance")

# -----------------------------------------------------------------------------
# Pick configs: fixed GPU per quantization mode
# -----------------------------------------------------------------------------

PICK_CONFIGS = {
    "int4-awq": {"gpu_search": "H100", "gpu_exclude": ["PCIE"], "trtllm_quant": "w4a16_awq"},
    "fp8":      {"gpu_search": "H200", "gpu_exclude": [],       "trtllm_quant": "fp8"},
}

# -----------------------------------------------------------------------------
# Mutable state
# -----------------------------------------------------------------------------

current_instance_id: Optional[int] = None
ssh_process: Optional[asyncio.subprocess.Process] = None
known_hosts_file: Optional[str] = None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def now() -> float:
    return time.time()


def vast_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {VAST_API_KEY}",
        "Content-Type": "application/json",
    }


def _normalize(s: str) -> List[str]:
    return re.sub(r"[-_]", " ", s.upper()).split()


def gpu_matches(offer_name: str, search: str, exclude: List[str]) -> bool:
    words = set(_normalize(offer_name))
    if not all(w in words for w in _normalize(search)):
        return False
    if exclude and any(w in words for w in _normalize(" ".join(exclude))):
        return False
    return True


def extract_price_per_hour(obj: Dict[str, Any]) -> Optional[float]:
    for path in [("dph_total",), ("discounted_dph_total",), ("dph_total_adj",), ("discounted_hourly",)]:
        cur: Any = obj
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and isinstance(cur, (int, float)):
            return float(cur)
    return None


def extract_public_ip(instance: Dict[str, Any]) -> Optional[str]:
    for key in ("public_ipaddr", "public_ip", "ssh_host", "host", "actual_ip"):
        val = instance.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def infer_running(instance: Dict[str, Any]) -> bool:
    for key in ("actual_status", "cur_state", "next_state", "intended_status", "status", "state"):
        val = instance.get(key)
        if isinstance(val, str):
            s = val.lower()
            if "running" in s or s == "up":
                return True
    return False


def instance_destroyed(instance: Dict[str, Any]) -> bool:
    for key in ("actual_status", "cur_state", "status", "state"):
        val = instance.get(key)
        if isinstance(val, str) and "destroy" in val.lower():
            return True
    return False


def get_ssh_host_and_port(instance: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    ip = extract_public_ip(instance)
    if not ip:
        return None
    ports: Dict[str, Any] = instance.get("ports") or {}
    mappings = ports.get("2222/tcp") or []
    host_port = None
    for m in mappings:
        if isinstance(m, dict) and m.get("HostIp") in ("0.0.0.0", ""):
            host_port = m.get("HostPort")
            break
    if not host_port and mappings:
        host_port = mappings[0].get("HostPort") if isinstance(mappings[0], dict) else None
    if not host_port:
        log.debug("No host port mapping yet for SSH (2222/tcp) on instance %s", instance.get("id"))
        return None
    return (ip, int(host_port))


async def check_tcp_port(ip: str, port: int) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(ip, port), timeout=5.0)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception:
        return False


async def fetch_host_pubkeys(client: httpx.AsyncClient, instance_id: int) -> Optional[List[str]]:
    r = await client.put(
        f"{VAST_API_BASE}/instances/request_logs/{instance_id}/",
        headers=vast_headers(),
        json={"tail": "100"},
    )
    if not r.is_success:
        return None
    result_url = r.json().get("result_url")
    if not result_url:
        return None
    await asyncio.sleep(3.0)
    r2 = await client.get(result_url)
    if not r2.is_success:
        return None
    lines = [l.strip() for l in r2.text.splitlines()]
    lines.reverse()
    keys: Optional[List[str]] = None
    for s in lines:
        if s == "===END HOST PUBLIC KEYS===":
            keys = []
        elif s == "===BEGIN HOST PUBLIC KEYS===":
            return keys
        elif keys is not None and s:
            keys.append(s)
    return None


def write_known_hosts(ip: str, host_port: int, keys: List[str]) -> str:
    fd, path = tempfile.mkstemp(suffix=".known_hosts")
    with os.fdopen(fd, "w") as f:
        for key in keys:
            f.write(f"[{ip}]:{host_port} {key}\n")
    return path


async def connect_instance(ip: str, host_port: int, known_hosts_path: str, model_name: str, trtllm_quant: str) -> asyncio.subprocess.Process:
    quant_flag = f'--quantization "{trtllm_quant}"' if trtllm_quant else ""
    script = SERVER_SH.format(model_name=model_name, quant_flag=quant_flag)
    cmd = [
        "ssh",
        "-L", f"{LOCAL_LLAMA_PORT}:127.0.0.1:8080",
        "-p", str(host_port),
        "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={known_hosts_path}",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3",
        f"fox@{ip}",
        "sh",
    ]
    log.info("Connecting to instance: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.PIPE)
    proc.stdin.write(script.encode())
    await proc.stdin.drain()
    proc.stdin.close()
    return proc


# -----------------------------------------------------------------------------
# Vast API wrappers
# -----------------------------------------------------------------------------

async def vast_show_instances(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    r = await client.get(f"{VAST_API_BASE}/instances/", headers=vast_headers())
    r.raise_for_status()
    return r.json().get("instances", [])


async def vast_show_instance(client: httpx.AsyncClient, instance_id: int) -> Dict[str, Any]:
    r = await client.get(f"{VAST_API_BASE}/instances/{instance_id}/", headers=vast_headers())
    if r.status_code == 404:
        raise RuntimeError(f"Instance {instance_id} not found")
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict):
        instances = payload.get("instances")
        if isinstance(instances, dict):
            return instances
        if isinstance(instances, list) and instances:
            return instances[0]
        if "id" in payload:
            return payload
    raise RuntimeError(f"Unexpected instance response: {list(payload.keys()) if isinstance(payload, dict) else payload}")


async def vast_manage_instance(client: httpx.AsyncClient, instance_id: int, new_state: str) -> None:
    r = await client.put(
        f"{VAST_API_BASE}/instances/{instance_id}/",
        headers=vast_headers(),
        json={"state": new_state},
    )
    r.raise_for_status()


async def vast_destroy_instance(client: httpx.AsyncClient, instance_id: int) -> None:
    r = await client.delete(f"{VAST_API_BASE}/instances/{instance_id}/", headers=vast_headers())
    r.raise_for_status()


async def vast_search_offers(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    body: Dict[str, Any] = {
        "limit": SEARCH_LIMIT,
        "type": "on-demand",
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "gpu_arch": {"eq": "nvidia"},
        "num_gpus": {"eq": 1},
        "reliability": {"gte": REQUIRE_RELIABILITY_GTE},
        "order": [["dph_total", "asc"]],
        "disable_bundling": True,
    }
    if PREFER_VERIFIED:
        body["verified"] = {"eq": True}

    r = await client.post(f"{VAST_API_BASE}/bundles", headers=vast_headers(), json=body)
    r.raise_for_status()
    payload = r.json()

    raw = payload.get("offers", [])
    if isinstance(raw, dict):
        raw = [raw]
    elif not isinstance(raw, list):
        raw = []

    out: List[Dict[str, Any]] = []
    for item in raw:
        offer = item.get("offers", item)
        if isinstance(offer, dict):
            out.append(offer)
    return out


async def vast_create_instance_from_offer(client: httpx.AsyncClient, offer: Dict[str, Any]) -> int:
    ask_id = offer.get("ask_contract_id") or offer.get("id")
    if not ask_id:
        raise RuntimeError("Offer missing ask ID")

    r = await client.put(
        f"{VAST_API_BASE}/asks/{ask_id}/",
        headers=vast_headers(),
        json={
            "disk": VAST_DISK_GB,
            "target_state": "running",
            "label": INSTANCE_LABEL,
            "cancel_unavail": True,
            "runtype": "args",
            "image": VAST_IMAGE,
            "env": {"-p 2222:2222": "1"},
        },
    )
    if r.status_code != 200:
        raise Exception(f"Failed to create instance from offer: {r.status_code} {r.text}")
    payload = r.json()
    new_contract = payload.get("new_contract")
    if not isinstance(new_contract, int):
        raise RuntimeError(f"Unexpected create-instance response: {payload}")
    return new_contract


# -----------------------------------------------------------------------------
# Selection logic
# -----------------------------------------------------------------------------

def rank_existing_instance(instance: Dict[str, Any]) -> Tuple[int, float, int]:
    running_score = 0 if infer_running(instance) else 1
    price = extract_price_per_hour(instance) or 999999.0
    return (running_score, price, int(instance.get("id", 1_000_000_000)))


def rank_offer(offer: Dict[str, Any]) -> Tuple[float, float, int]:
    price = extract_price_per_hour(offer) or 999999.0
    reliability = float(offer.get("reliability", 0.0) or 0.0)
    return (price, -reliability, int(offer.get("id", 1_000_000_000)))


async def choose_or_create_instance(
    client: httpx.AsyncClient,
    offers: List[Dict[str, Any]],
    gpu_search: str,
    gpu_exclude: List[str],
) -> int:
    # Reuse existing matching instance if any.
    instances = await vast_show_instances(client)
    matches = [
        i for i in instances
        if gpu_matches(str(i.get("gpu_name") or ""), gpu_search, gpu_exclude)
        and not instance_destroyed(i)
    ]
    if matches:
        matches.sort(key=rank_existing_instance)
        chosen = matches[0]
        iid = int(chosen["id"])
        log.info(
            "Reusing existing instance id=%s gpu=%s status=%s",
            iid, chosen.get("gpu_name"),
            chosen.get("actual_status") or chosen.get("cur_state") or chosen.get("status"),
        )
        return iid

    # Find cheapest matching offer.
    candidates = [
        o for o in offers
        if gpu_matches(str(o.get("gpu_name") or ""), gpu_search, gpu_exclude)
    ]
    if not candidates:
        raise RuntimeError(f"No matching '{gpu_search}' offers found on Vast")

    candidates.sort(key=rank_offer)
    best = candidates[0]
    best_price = extract_price_per_hour(best)
    if best_price is None:
        raise RuntimeError("Matching offer found but price could not be determined")
    if best_price > MAX_HOURLY_PRICE:
        raise RuntimeError(
            f"Cheapest '{gpu_search}' offer costs ${best_price:.2f}/hr, "
            f"above MAX_HOURLY_PRICE=${MAX_HOURLY_PRICE:.2f}/hr"
        )

    log.info(
        "Creating instance from offer id=%s gpu=%s price=$%.2f/hr",
        best.get("id"), best.get("gpu_name"), best_price,
    )
    return await vast_create_instance_from_offer(client, best)


async def ensure_instance_ready(pick_mode: str) -> None:
    global current_instance_id, ssh_process, known_hosts_file

    cfg = PICK_CONFIGS[pick_mode]
    gpu_search: str = cfg["gpu_search"]
    gpu_exclude: List[str] = cfg["gpu_exclude"]
    trtllm_quant: str = cfg["trtllm_quant"]

    async with httpx.AsyncClient(timeout=60.0) as client:
        log.info("Fetching Vast.ai GPU offers")
        offers = await vast_search_offers(client)

        instance_id = await choose_or_create_instance(client, offers, gpu_search, gpu_exclude)
        current_instance_id = instance_id

        instance = await vast_show_instance(client, instance_id)
        if not infer_running(instance):
            log.info("Starting instance %s", instance_id)
            await vast_manage_instance(client, instance_id, "running")

        deadline = now() + STARTUP_TIMEOUT
        last_err: Optional[Exception] = None
        ip: Optional[str] = None
        host_port: Optional[int] = None

        while now() < deadline:
            try:
                instance = await vast_show_instance(client, instance_id)
                status = (
                    instance.get("actual_status")
                    or instance.get("cur_state")
                    or instance.get("status")
                    or "unknown"
                )
                log.info("Instance %s status: %s", instance_id, status)
                if infer_running(instance):
                    conn = get_ssh_host_and_port(instance)
                    if not conn:
                        log.info("Waiting for port mapping on instance %s...", instance_id)
                        await asyncio.sleep(HEALTHCHECK_INTERVAL)
                        continue
                    ip, host_port = conn
                    if not await check_tcp_port(ip, host_port):
                        log.info("SSH port not yet reachable at %s:%s, retrying...", ip, host_port)
                        ip = None
                        host_port = None
                        await asyncio.sleep(HEALTHCHECK_INTERVAL)
                        continue
                    break
            except Exception as exc:
                last_err = exc
                log.warning("Error polling instance %s: %s", instance_id, exc)

            await asyncio.sleep(HEALTHCHECK_INTERVAL)

        if not ip or not host_port:
            detail = f"Timed out waiting for instance {instance_id} / SSH readiness"
            if last_err:
                detail += f": {last_err}"
            raise RuntimeError(detail)

        log.info("Fetching host public keys for instance %s", instance_id)
        keys = await fetch_host_pubkeys(client, instance_id)
        if not keys:
            if keys is None:
                raise RuntimeError(f"Failed to fetch host public keys for instance {instance_id}: Not in log")
            else:
                raise RuntimeError(f"Failed to fetch host public keys for instance {instance_id}: Empty keys block")
        known_hosts_file = write_known_hosts(ip, host_port, keys)
        log.info("Host keys written to %s", known_hosts_file)
        ssh_process = await connect_instance(ip, host_port, known_hosts_file, MODEL_NAME, trtllm_quant)
        log.info("Connected to instance %s", instance_id)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start a Vast.ai instance and serve a HuggingFace model with TensorRT-LLM.")
    parser.add_argument(
        "--pick", required=True, choices=list(PICK_CONFIGS),
        help="Quantization mode to use.")
    args = parser.parse_args()

    await ensure_instance_ready(args.pick)
    log.info("Instance ready. Press Ctrl+C to stop.")
    try:
        if ssh_process is not None:
            await ssh_process.wait()
        else:
            await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        if ssh_process is not None:
            log.info("Terminating SSH tunnel")
            ssh_process.terminate()
            try:
                await asyncio.wait_for(ssh_process.wait(), timeout=5.0)
            except Exception:
                ssh_process.kill()
        if known_hosts_file is not None:
            os.unlink(known_hosts_file)
        if current_instance_id is not None:
            async with httpx.AsyncClient(timeout=60.0) as client:
                if INSTANCE_ACTION == "destroy":
                    log.info("Destroying instance %s", current_instance_id)
                    await vast_destroy_instance(client, current_instance_id)
                else:
                    log.info("Stopping instance %s", current_instance_id)
                    await vast_manage_instance(client, current_instance_id, "stopped")


if __name__ == "__main__":
    asyncio.run(main())
