#!/usr/bin/env python3
"""
vast_instance.py

Starts a Vast.ai instance and opens an SSH tunnel, then waits until interrupted.

Behavior:
- Fetches model architecture from HuggingFace to determine VRAM requirements
- Queries Vast.ai for available GPUs and picks the best by tok/s/$ for the
  chosen quantization mode (--pick int4-awq | fp8)
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
    export HF_TOKEN="hf_..."        # optional, for gated models
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
# Config
# -----------------------------------------------------------------------------

VAST_API_KEY = os.environ.get("VAST_API_KEY", "").strip()
VAST_API_BASE = os.environ.get("VAST_API_BASE", "https://console.vast.ai/api/v0").rstrip("/")
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

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
# GPU targets and quantization
# -----------------------------------------------------------------------------

OVERHEAD_GB   = 3.0   # framework + activations
BW_EFFICIENCY = 0.80  # practical fraction of theoretical peak bandwidth

# GPU names verified against live Vast.ai offer data.
GPU_TARGETS = [
    {"label": "B200",         "vram_gb": 192, "bw_tbs": 8.0,   "fp8": True,  "search": "B200",         "exclude": []},
    {"label": "H200 NVL",     "vram_gb": 141, "bw_tbs": 4.8,   "fp8": True,  "search": "H200 NVL",     "exclude": []},
    {"label": "H200",         "vram_gb": 141, "bw_tbs": 4.8,   "fp8": True,  "search": "H200",         "exclude": ["NVL"]},
    {"label": "H100 SXM",     "vram_gb": 80,  "bw_tbs": 3.35,  "fp8": True,  "search": "H100 SXM",     "exclude": ["NVL", "PCIE"]},
    {"label": "H100 NVL",     "vram_gb": 94,  "bw_tbs": 3.9,   "fp8": True,  "search": "H100 NVL",     "exclude": []},
    {"label": "H100 PCIe",    "vram_gb": 80,  "bw_tbs": 2.0,   "fp8": True,  "search": "H100 PCIE",    "exclude": ["SXM", "NVL"]},
    {"label": "A100 SXM4",    "vram_gb": 80,  "bw_tbs": 2.0,   "fp8": False, "search": "A100 SXM4",    "exclude": ["PCIE"]},
    {"label": "A100 PCIe",    "vram_gb": 80,  "bw_tbs": 2.0,   "fp8": False, "search": "A100 PCIE",    "exclude": ["SXM"]},
    {"label": "RTX PRO 6000", "vram_gb": 96,  "bw_tbs": 0.576, "fp8": False, "search": "RTX PRO 6000", "exclude": []},
]

# Ordered best→worst quality.
# bpp = bytes per weight parameter; fp8_kv = use FP8 KV cache (Hopper+ only).
# trtllm_quant = value passed to trtllm-serve --quantization (empty = default BF16).
QUANTS = [
    {"name": "BF16",     "bpp": 2.0, "requires_fp8": False, "fp8_kv": False, "trtllm_quant": ""},
    {"name": "FP8",      "bpp": 1.0, "requires_fp8": True,  "fp8_kv": True,  "trtllm_quant": "fp8"},
    {"name": "INT8",     "bpp": 1.0, "requires_fp8": False, "fp8_kv": False, "trtllm_quant": "int8"},
    {"name": "INT4-AWQ", "bpp": 0.5, "requires_fp8": False, "fp8_kv": False, "trtllm_quant": "w4a16_awq"},
]

# -----------------------------------------------------------------------------
# Mutable state
# -----------------------------------------------------------------------------

state_lock = asyncio.Lock()

current_instance_id: Optional[int] = None
ssh_process: Optional[asyncio.subprocess.Process] = None
known_hosts_file: Optional[str] = None

# -----------------------------------------------------------------------------
# GPU / quant helpers
# -----------------------------------------------------------------------------

def _normalize(s: str) -> List[str]:
    return re.sub(r"[-_]", " ", s.upper()).split()


def gpu_matches(offer_name: str, search: str, exclude: List[str]) -> bool:
    """True if every word in search appears in the GPU name and no exclude word does."""
    words = set(_normalize(offer_name))
    if not all(w in words for w in _normalize(search)):
        return False
    if exclude and any(w in words for w in _normalize(" ".join(exclude))):
        return False
    return True


def kv_cache_gb(p: Dict[str, Any], kv_bytes_per_elem: int) -> float:
    """Full-context KV cache size in GB."""
    b = 2 * p["num_layers"] * p["num_kv_heads"] * p["head_dim"] * p["context_len"] * kv_bytes_per_elem
    return b / 1e9


def best_quant_for_gpu(p: Dict[str, Any], gpu: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Highest-quality quant that fits in gpu VRAM with the full KV cache."""
    if p["num_params"] is None:
        return None
    for q in QUANTS:
        if q["requires_fp8"] and not gpu["fp8"]:
            continue
        model_gb = p["num_params"] * q["bpp"] / 1e9
        kv_elem  = 1 if (q["fp8_kv"] and gpu["fp8"]) else 2
        kv_gb    = kv_cache_gb(p, kv_elem)
        total_gb = model_gb + kv_gb + OVERHEAD_GB
        if total_gb <= gpu["vram_gb"]:
            return {**q, "model_gb": model_gb, "kv_gb": kv_gb, "total_gb": total_gb}
    return None


def theoretical_tps(p: Dict[str, Any], gpu: Dict[str, Any], q: Dict[str, Any]) -> float:
    """Theoretical tokens/s for batch=1 autoregressive decoding."""
    return (gpu["bw_tbs"] * 1e12 / (p["num_params"] * q["bpp"])) * BW_EFFICIENCY

# -----------------------------------------------------------------------------
# HuggingFace helpers
# -----------------------------------------------------------------------------

def hf_headers() -> Dict[str, str]:
    h: Dict[str, str] = {}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    return h


async def fetch_model_info(client: httpx.AsyncClient) -> Dict[str, Any]:
    config_url = f"https://huggingface.co/{MODEL_NAME}/resolve/main/config.json"
    r = await client.get(config_url, headers=hf_headers(), follow_redirects=True)
    r.raise_for_status()
    config = r.json()

    r2 = await client.get(f"https://huggingface.co/api/models/{MODEL_NAME}", headers=hf_headers())
    r2.raise_for_status()
    meta = r2.json()

    return {"config": config, "meta": meta}


def parse_model_params(info: Dict[str, Any]) -> Dict[str, Any]:
    config = info["config"]
    meta   = info["meta"]

    num_params: Optional[int] = None
    sf = meta.get("safetensors") or {}
    if isinstance(sf, dict):
        num_params = sf.get("total")

    if num_params is None:
        for tag in meta.get("tags", []):
            t = tag.lower().strip()
            if t.endswith("b") and t[:-1].replace(".", "").isdigit():
                num_params = int(float(t[:-1]) * 1_000_000_000)
                break

    num_layers    = config.get("num_hidden_layers", 32)
    num_attn      = config.get("num_attention_heads", 32)
    num_kv_heads  = config.get("num_key_value_heads", num_attn)
    hidden_size   = config.get("hidden_size", 4096)
    head_dim      = config.get("head_dim", hidden_size // num_attn)
    context_len   = config.get("max_position_embeddings", 32768)

    return {
        "num_params":   num_params,
        "num_layers":   num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim":     head_dim,
        "context_len":  context_len,
    }

# -----------------------------------------------------------------------------
# GPU picker
# -----------------------------------------------------------------------------

def _cheapest_price(offers: List[Dict[str, Any]], search: str, exclude: List[str]) -> Optional[float]:
    prices = []
    for o in offers:
        if not gpu_matches(str(o.get("gpu_name") or ""), search, exclude):
            continue
        for key in ("dph_total", "discounted_dph_total", "dph_total_adj", "discounted_hourly"):
            v = o.get(key)
            if isinstance(v, (int, float)) and v > 0:
                prices.append(float(v))
                break
    return min(prices) if prices else None


def build_gpu_rows(model_params: Dict[str, Any], offers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for gpu in GPU_TARGETS:
        q     = best_quant_for_gpu(model_params, gpu)
        if q is None:
            continue
        price = _cheapest_price(offers, gpu["search"], gpu["exclude"])
        tps   = theoretical_tps(model_params, gpu, q)
        tpd   = (tps / price) if price else None
        rows.append({
            "gpu_target":    gpu,
            "gpu":           gpu["label"],
            "quant":         q["name"],
            "trtllm_quant":  q["trtllm_quant"],
            "tps":           tps,
            "price":         price,
            "tpd":           tpd,
        })
    return rows


def pick_best_int4awq(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the GPU row with the best tok/s/$ running INT4-AWQ quantization."""
    candidates = [r for r in rows if r["quant"] == "INT4-AWQ" and r["tpd"] is not None]
    return max(candidates, key=lambda r: r["tpd"], default=None)


def pick_best_fp8(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the GPU row with the best tok/s/$ running FP8 quantization."""
    candidates = [r for r in rows if r["quant"] == "FP8" and r["tpd"] is not None]
    return max(candidates, key=lambda r: r["tpd"], default=None)

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


async def connect_instance(ip: str, host_port: int, known_hosts_path: str) -> asyncio.subprocess.Process:
    cmd = [
        "ssh",
        "-L", f"{LLAMA_PORT}:127.0.0.1:8080",
        "-p", str(host_port),
        "-o", "ConnectTimeout=5",
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
    """Broad search for all high-VRAM GPU offers (filtered to target type locally)."""
    body: Dict[str, Any] = {
        "limit": SEARCH_LIMIT,
        "type": "on-demand",
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "gpu_arch": {"eq": "nvidia"},
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": 70000},
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


async def vast_create_instance_from_offer(
    client: httpx.AsyncClient,
    offer: Dict[str, Any],
    trtllm_quant: str,
) -> int:
    ask_id = offer.get("ask_contract_id") or offer.get("id")
    if not ask_id:
        raise RuntimeError("Offer missing ask ID")

    env: Dict[str, str] = {f"-p {VAST_SSH_PORT}:{VAST_SSH_PORT}": "1"}
    env["MODEL_NAME"] = MODEL_NAME
    env["TRTLLM_QUANT"] = trtllm_quant

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

def existing_instance_matches(instance: Dict[str, Any], gpu_target: Dict[str, Any]) -> bool:
    gpu_name = str(instance.get("gpu_name") or "")
    if not gpu_matches(gpu_name, gpu_target["search"], gpu_target["exclude"]):
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


async def choose_or_create_instance(
    client: httpx.AsyncClient,
    offers: List[Dict[str, Any]],
    gpu_target: Dict[str, Any],
    trtllm_quant: str,
) -> int:
    # 1) Reuse existing matching instance if any.
    instances = await vast_show_instances(client)
    matches = [i for i in instances if existing_instance_matches(i, gpu_target)]
    if matches:
        matches.sort(key=rank_existing_instance)
        chosen = matches[0]
        iid = int(chosen["id"])
        log.info(
            "Reusing existing instance id=%s gpu=%s status=%s",
            iid,
            chosen.get("gpu_name"),
            chosen.get("actual_status") or chosen.get("cur_state") or chosen.get("status"),
        )
        return iid

    # 2) Filter pre-fetched offers to the selected GPU type.
    candidates = [
        o for o in offers
        if gpu_matches(str(o.get("gpu_name") or ""), gpu_target["search"], gpu_target["exclude"])
    ]
    if not candidates:
        raise RuntimeError(f"No matching {gpu_target['label']} offers found on Vast")

    candidates.sort(key=rank_offer)
    best = candidates[0]
    best_price = extract_price_per_hour(best)
    if best_price is None:
        raise RuntimeError("Matching offer found but price could not be determined")

    if best_price > MAX_HOURLY_PRICE:
        raise RuntimeError(
            f"Cheapest {gpu_target['label']} offer costs ${best_price:.2f}/hr, "
            f"above MAX_HOURLY_PRICE=${MAX_HOURLY_PRICE:.2f}/hr"
        )

    log.info(
        "Creating instance from cheapest offer id=%s ask_id=%s gpu=%s price=$%.2f/hr",
        best.get("id"),
        best.get("ask_contract_id"),
        best.get("gpu_name"),
        best_price,
    )
    return await vast_create_instance_from_offer(client, best, trtllm_quant)


async def ensure_instance_ready(pick_mode: str) -> None:
    global current_instance_id, ssh_process, known_hosts_file

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Fetch model architecture from HuggingFace.
        log.info("Fetching model info for %s", MODEL_NAME)
        info = await fetch_model_info(client)
        model_params = parse_model_params(info)
        if model_params["num_params"] is None:
            raise RuntimeError("Could not determine parameter count from HuggingFace metadata")
        log.info(
            "Model: %.1fB params | %d layers | %d KV heads × %dd | %d token context",
            model_params["num_params"] / 1e9,
            model_params["num_layers"],
            model_params["num_kv_heads"],
            model_params["head_dim"],
            model_params["context_len"],
        )

        # 2. Fetch all high-VRAM Vast offers (used for both GPU selection and instance creation).
        log.info("Fetching Vast.ai GPU offers")
        offers = await vast_search_offers(client)

        # 3. Pick the best GPU for the requested quantization mode.
        rows = build_gpu_rows(model_params, offers)
        if pick_mode == "int4-awq":
            picked = pick_best_int4awq(rows)
        else:
            picked = pick_best_fp8(rows)

        if picked is None:
            raise RuntimeError(
                f"No suitable GPU found for {pick_mode} — no priced offers available on Vast"
            )

        gpu_target   = picked["gpu_target"]
        trtllm_quant = picked["trtllm_quant"]
        log.info(
            "Selected: %s  quant=%s  ~%.0f tok/s  $%.2f/hr  ~%.0f tok/s/$",
            picked["gpu"], picked["quant"],
            picked["tps"] or 0, picked["price"] or 0, picked["tpd"] or 0,
        )

        # 4. Find or create an instance of the selected GPU type.
        instance_id = await choose_or_create_instance(client, offers, gpu_target, trtllm_quant)
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
        ssh_process = await connect_instance(ip, host_port, known_hosts_file)
        log.info("Connected to instance %s", instance_id)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start a Vast.ai instance and serve a HuggingFace model with TensorRT-LLM.")
    parser.add_argument(
        "--pick", required=True, choices=["int4-awq", "fp8"],
        help="Quantization strategy — selects the GPU with the best tok/s/$ for that mode.")
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
