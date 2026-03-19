#!/usr/bin/env python3
"""
vast_instance.py

Starts a Vast.ai instance and opens an SSH tunnel, then waits until interrupted.

Behavior:
- Reuse an existing Vast instance if its GPU is in the RTX PRO 6000 family
  (base / S / WS all accepted)
- Otherwise find the cheapest matching offer on Vast
- Create a new instance only if the cheapest offer is <= MAX_HOURLY_PRICE
- Start the instance and wait for SSH port to become available
- Open an SSH local-port-forward tunnel (localhost:6666 -> remote:6666)
- Stop or destroy the instance on exit

Requirements:
    pip install httpx

Example:
    export VAST_API_KEY="..."

    # Required: model to load in llama.cpp
    export LLAMA_MODEL_URL="https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"

    # Optional HF token for gated/private models
    export HF_TOKEN="hf_..."

    export MAX_HOURLY_PRICE="1.00"
    export INSTANCE_ACTION="stop"   # or "destroy"

    python vast_instance.py
"""

import asyncio
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

VAST_API_KEY = os.environ.get("VAST_API_KEY", "").strip()
VAST_API_BASE = os.environ.get("VAST_API_BASE", "https://console.vast.ai/api/v0").rstrip("/")

# SSH tunnel constants
VAST_SSH_PORT = 2222
LLAMA_PORT = 6969
SSH_USER = "fox"

STARTUP_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT", "900"))
HEALTHCHECK_INTERVAL = float(os.environ.get("HEALTHCHECK_INTERVAL", "3"))

# Instance selection / creation
MAX_HOURLY_PRICE = float(os.environ.get("MAX_HOURLY_PRICE", "1.00"))
PREFER_VERIFIED = os.environ.get("PREFER_VERIFIED", "true").lower() == "true"
REQUIRE_RELIABILITY_GTE = float(os.environ.get("REQUIRE_RELIABILITY_GTE", "0.95"))
SEARCH_LIMIT = int(os.environ.get("SEARCH_LIMIT", "50"))

VAST_IMAGE = os.environ.get("VAST_IMAGE", "ghcr.io/doridian/vastimage/vastimage:latest").strip()

# Create-instance config
VAST_DISK_GB = float(os.environ.get("VAST_DISK_GB", "100"))
INSTANCE_LABEL = os.environ.get("INSTANCE_LABEL", "vastimage-controlled").strip()

# llama.cpp env options
LLAMA_MODEL_URL = os.environ.get("LLAMA_MODEL_URL", "").strip()

# Shutdown action
INSTANCE_ACTION = os.environ.get("INSTANCE_ACTION", "stop").strip().lower()  # stop | destroy

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

if not VAST_API_KEY:
    raise RuntimeError("VAST_API_KEY is required")
if not LLAMA_MODEL_URL:
    raise RuntimeError("LLAMA_MODEL_URL is required")
if INSTANCE_ACTION not in {"stop", "destroy"}:
    raise RuntimeError("INSTANCE_ACTION must be 'stop' or 'destroy'")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("vast-instance")

# -----------------------------------------------------------------------------
# Mutable state
# -----------------------------------------------------------------------------

state_lock = asyncio.Lock()

current_instance_id: Optional[int] = None
instance_ready = False
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


def normalize_gpu_name(name: str) -> str:
    s = name.upper().strip()
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def gpu_name_matches_rtx_pro_6000_family(name: str) -> bool:
    """
    Accept:
      - RTX PRO 6000
      - RTX PRO 6000 S
      - RTX PRO 6000 WS
      - longer marketplace spellings containing those forms

    Reject:
      - MAX-Q
      - unrelated cards
    """
    s = normalize_gpu_name(name)
    if "RTX PRO 6000" not in s:
        return False
    if "MAX Q" in s or "MAXQ" in s:
        return False
    return True


def extract_price_per_hour(obj: Dict[str, Any]) -> Optional[float]:
    """
    Vast payloads can expose price under different keys depending on endpoint.
    Prefer total hourly cost when present.
    """
    candidates = [
        ("search", "totalHour"),
        ("instance", "totalHour"),
        ("search", "discountedTotalPerHour"),
        ("instance", "discountedTotalPerHour"),
        ("totalHour",),
        ("discountedTotalPerHour",),
        ("dph_total",),
        ("discounted_dph_total",),
        ("dph_total_adj",),
        ("discounted_hourly",),
    ]
    for path in candidates:
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
    mappings = ports.get(f"{VAST_SSH_PORT}/tcp") or []
    host_port = None
    for m in mappings:
        if isinstance(m, dict) and m.get("HostIp") in ("0.0.0.0", ""):
            host_port = m.get("HostPort")
            break
    if not host_port and mappings:
        host_port = mappings[0].get("HostPort") if isinstance(mappings[0], dict) else None
    if not host_port:
        log.debug("No host port mapping yet for %s/tcp on instance %s", VAST_SSH_PORT, instance.get("id"))
        return None
    return (ip, int(host_port))


async def fetch_host_pubkeys(client: httpx.AsyncClient, instance_id: int) -> Optional[List[str]]:
    r = await client.put(
        f"{VAST_API_BASE}/instances/request_logs/{instance_id}/",
        headers=vast_headers(),
        json={"tail": "1000"},
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
    in_block = False
    keys: List[str] = []
    for line in r2.text.splitlines():
        s = line.strip()
        if s == "===BEGIN HOST PUBLIC KEYS===":
            in_block = True
        elif s == "===END HOST PUBLIC KEYS===":
            return keys
        elif in_block and s:
            parts = s.split()
            if len(parts) >= 2:
                keys.append(f"{parts[0]} {parts[1]}")
    return None


def write_known_hosts(ip: str, host_port: int, keys: List[str]) -> str:
    fd, path = tempfile.mkstemp(suffix=".known_hosts")
    with os.fdopen(fd, "w") as f:
        for key in keys:
            f.write(f"[{ip}]:{host_port} {key}\n")
    return path


async def connect_instance(ip: str, host_port: int, known_hosts_path: str) -> asyncio.subprocess.Process:
    cmd = [
        "ssh",
        "-L", f"{LLAMA_PORT}:127.0.0.1:8080",
        "-p", str(host_port),
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={known_hosts_path}",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3",
        f"{SSH_USER}@{ip}",
        "exec /server.sh",
    ]
    log.info("Connecting to instance: %s", " ".join(cmd))
    return await asyncio.create_subprocess_exec(*cmd)


# -----------------------------------------------------------------------------
# Vast API wrappers
# -----------------------------------------------------------------------------

async def vast_show_instances(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    r = await client.get(f"{VAST_API_BASE}/instances/", headers=vast_headers())
    r.raise_for_status()
    payload = r.json()
    return payload.get("instances", [])


async def vast_show_instance(client: httpx.AsyncClient, instance_id: int) -> Dict[str, Any]:
    r = await client.get(f"{VAST_API_BASE}/instances/{instance_id}/", headers=vast_headers())
    if r.status_code == 404:
        raise RuntimeError(f"Instance {instance_id} not found")
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict):
        instances = payload.get("instances")
        # GET /instances/{id}/ returns {"instances": <dict>}  (single instance as dict)
        if isinstance(instances, dict):
            return instances
        # GET /instances/ returns {"instances": [<dict>, ...]}
        if isinstance(instances, list) and instances:
            return instances[0]
        # bare object fallback
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
    """
    Broad search, then local fuzzy matching on GPU name.
    """
    body: Dict[str, Any] = {
        "limit": SEARCH_LIMIT,
        "type": "on-demand",
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "gpu_arch": {"eq": "nvidia"},
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": 96000},
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

    env: Dict[str, str] = {f"-p {VAST_SSH_PORT}:{VAST_SSH_PORT}": "1"}
    env["LLAMA_MODEL_URL"] = LLAMA_MODEL_URL

    body: Dict[str, Any] = {
        "disk": VAST_DISK_GB,
        "target_state": "running",
        "label": INSTANCE_LABEL,
        "cancel_unavail": True,
        "runtype": "args",
        "image": VAST_IMAGE,
        "env": env,
    }

    r = await client.put(
        f"{VAST_API_BASE}/asks/{ask_id}/",
        headers=vast_headers(),
        json=body,
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

def existing_instance_matches(instance: Dict[str, Any]) -> bool:
    gpu_name = str(instance.get("gpu_name") or "")
    if not gpu_name_matches_rtx_pro_6000_family(gpu_name):
        return False
    if instance_destroyed(instance):
        return False
    return True


def rank_existing_instance(instance: Dict[str, Any]) -> Tuple[int, float, int]:
    """
    Prefer:
    1. already running
    2. lower price if available
    3. lower instance id
    """
    running_score = 0 if infer_running(instance) else 1
    price = extract_price_per_hour(instance)
    if price is None:
        price = 999999.0
    iid = int(instance.get("id", 1_000_000_000))
    return (running_score, price, iid)


def rank_offer(offer: Dict[str, Any]) -> Tuple[float, float, int]:
    price = extract_price_per_hour(offer)
    if price is None:
        price = 999999.0
    reliability = float(offer.get("reliability", 0.0) or 0.0)
    oid = int(offer.get("id", 1_000_000_000))
    return (price, -reliability, oid)


async def choose_or_create_instance(client: httpx.AsyncClient) -> int:
    # 1) Reuse existing matching instance if any.
    instances = await vast_show_instances(client)
    matches = [i for i in instances if existing_instance_matches(i)]
    if matches:
        matches.sort(key=rank_existing_instance)
        chosen = matches[0]
        iid = int(chosen["id"])
        print(chosen)
        log.info(
            "Reusing existing instance id=%s gpu=%s status=%s",
            iid,
            chosen.get("gpu_name"),
            chosen.get("actual_status") or chosen.get("cur_state") or chosen.get("status"),
        )
        return iid

    # 2) Search cheapest matching offer.
    offers = await vast_search_offers(client)
    candidates = [o for o in offers if gpu_name_matches_rtx_pro_6000_family(str(o.get("gpu_name") or ""))]
    if not candidates:
        raise RuntimeError("No matching RTX PRO 6000 / S / WS offers found on Vast")

    candidates.sort(key=rank_offer)
    best = candidates[0]
    best_price = extract_price_per_hour(best)
    if best_price is None:
        raise RuntimeError("Matching offer found but price could not be determined")

    if best_price > MAX_HOURLY_PRICE:
        raise RuntimeError(
            f"Cheapest matching RTX PRO 6000-family offer costs ${best_price:.2f}/hr, "
            f"above MAX_HOURLY_PRICE=${MAX_HOURLY_PRICE:.2f}/hr"
        )

    log.info(
        "Creating instance from cheapest offer id=%s ask_id=%s gpu=%s price=$%.2f/hr",
        best.get("id"),
        best.get("ask_contract_id"),
        best.get("gpu_name"),
        best_price,
    )
    return await vast_create_instance_from_offer(client, best)


async def ensure_instance_ready() -> None:
    global current_instance_id, instance_ready, ssh_process, known_hosts_file

    async with httpx.AsyncClient(timeout=60.0) as client:
        instance_id = await choose_or_create_instance(client)
        current_instance_id = instance_id

        instance = await vast_show_instance(client, instance_id)
        if not infer_running(instance):
            log.info("Starting instance %s", instance_id)
            await vast_manage_instance(client, instance_id, "running")

        deadline = now() + STARTUP_TIMEOUT
        last_err: Optional[Exception] = None

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
                    log.info("Fetching host public keys for instance %s", instance_id)
                    keys = await fetch_host_pubkeys(client, instance_id)
                    if not keys:
                        log.info("Host public keys not yet in logs, retrying...")
                        await asyncio.sleep(HEALTHCHECK_INTERVAL)
                        continue
                    known_hosts_file = write_known_hosts(ip, host_port, keys)
                    log.info("Host keys written to %s", known_hosts_file)
                    ssh_process = await connect_instance(ip, host_port, known_hosts_file)
                    instance_ready = True
                    log.info("Connected to instance %s", instance_id)
                    return
            except Exception as exc:
                last_err = exc
                log.warning("Error polling instance %s: %s", instance_id, exc)

            await asyncio.sleep(HEALTHCHECK_INTERVAL)

        detail = f"Timed out waiting for instance {instance_id} / SSH readiness"
        if last_err:
            detail += f": {last_err}"
        raise RuntimeError(detail)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

async def main() -> None:
    await ensure_instance_ready()
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
