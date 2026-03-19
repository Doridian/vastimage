#!/usr/bin/env python3
"""
vast_roo_gateway.py

OpenAI-compatible gateway for:

  Roo Code / VS Code
          ↓
     this gateway
          ↓
   Vast.ai instance
     running llama.cpp

Behavior:
- Reuse an existing Vast instance if its GPU is in the RTX PRO 6000 family
  (base / S / WS all accepted)
- Otherwise find the cheapest matching offer on Vast
- Create a new instance only if the cheapest offer is <= MAX_HOURLY_PRICE
- Start the instance on demand
- Wait for remote llama.cpp server to become healthy
- Proxy /v1/* requests to the remote llama.cpp server
- Stop or destroy the instance after an idle timeout

Requirements:
    pip install fastapi "uvicorn[standard]" httpx

Example:
    export VAST_API_KEY="..."
    export GATEWAY_API_KEY="local-secret"

    # Required: model to load in llama.cpp
    export LLAMA_MODEL_URL="https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"

    # Optional extra llama-server args
    export LLAMA_ARGS="--ctx-size 8192"

    # Optional HF token for gated/private models
    export HF_TOKEN="hf_..."

    export MAX_HOURLY_PRICE="1.00"
    export INSTANCE_ACTION="stop"   # or "destroy"
    export IDLE_SECONDS="900"

    uvicorn vast_roo_gateway:app --host 127.0.0.1 --port 8089

Roo Code:
- Provider: OpenAI Compatible
- Base URL: http://127.0.0.1:8089/v1
- API Key: same as GATEWAY_API_KEY
- Model: whatever LLAMA_MODEL_URL is set to
"""

import asyncio
import logging
import os
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse, StreamingResponse

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

VAST_API_KEY = os.environ.get("VAST_API_KEY", "").strip()
GATEWAY_API_KEY = os.environ.get("GATEWAY_API_KEY", "").strip()

VAST_API_BASE = os.environ.get("VAST_API_BASE", "https://console.vast.ai/api/v0").rstrip("/")

# Remote llama.cpp endpoint info
VAST_LLAMA_PORT = 8080
# Optional manual override — set this to a CF tunnel URL or any base URL and
# the gateway will use it directly without touching the ports field.
# e.g. REMOTE_BASE_URL="https://keith-fascinating-nearby-decide.trycloudflare.com"
REMOTE_BASE_URL = os.environ.get("REMOTE_BASE_URL", "").strip()
REMOTE_SCHEME = os.environ.get("REMOTE_SCHEME", "http").strip()
HEALTHCHECK_PATH = os.environ.get("HEALTHCHECK_PATH", "/v1/models")
STARTUP_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT", "900"))
HEALTHCHECK_INTERVAL = float(os.environ.get("HEALTHCHECK_INTERVAL", "3"))

# Instance selection / creation
MAX_HOURLY_PRICE = float(os.environ.get("MAX_HOURLY_PRICE", "1.00"))
PREFER_VERIFIED = os.environ.get("PREFER_VERIFIED", "true").lower() == "true"
REQUIRE_RELIABILITY_GTE = float(os.environ.get("REQUIRE_RELIABILITY_GTE", "0.95"))
SEARCH_LIMIT = int(os.environ.get("SEARCH_LIMIT", "50"))

VAST_IMAGE = os.environ.get("VAST_IMAGE", "ghcr.io/doridian/vastimage/vastimage:latest").strip()

# Create-instance config
VAST_DISK_GB = float(os.environ.get("VAST_DISK_GB", "200"))
INSTANCE_LABEL = os.environ.get("INSTANCE_LABEL", "roo-vast-rtx-pro-6000").strip()

# llama.cpp env options
LLAMA_MODEL_URL = os.environ.get("LLAMA_MODEL_URL", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

# Idle shutdown
IDLE_SECONDS = int(os.environ.get("IDLE_SECONDS", "900"))
INSTANCE_ACTION = os.environ.get("INSTANCE_ACTION", "stop").strip().lower()  # stop | destroy

# Proxy timeout
PROXY_TIMEOUT = float(os.environ.get("PROXY_TIMEOUT", "1800"))

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

if not VAST_API_KEY:
    raise RuntimeError("VAST_API_KEY is required")
if not GATEWAY_API_KEY:
    raise RuntimeError("GATEWAY_API_KEY is required")
if not LLAMA_MODEL_URL:
    raise RuntimeError("LLAMA_MODEL_URL is required")
if INSTANCE_ACTION not in {"stop", "destroy"}:
    raise RuntimeError("INSTANCE_ACTION must be 'stop' or 'destroy'")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("vast-roo-gateway")

app = FastAPI(title="Vast Roo Gateway")

# -----------------------------------------------------------------------------
# Mutable state
# -----------------------------------------------------------------------------

state_lock = asyncio.Lock()
last_used_ts = 0.0
inflight_requests = 0

current_instance_id: Optional[int] = None
current_remote_base_url: Optional[str] = None
instance_ready = False

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


def build_remote_base_direct(instance: Dict[str, Any]) -> Optional[str]:
    ip = extract_public_ip(instance)
    if not ip:
        return None
    # ports = {"8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "26096"}, ...], ...}
    ports: Dict[str, Any] = instance.get("ports") or {}
    mappings = ports.get(f"{VAST_LLAMA_PORT}/tcp") or []
    host_port = None
    for m in mappings:
        if isinstance(m, dict) and m.get("HostIp") in ("0.0.0.0", ""):
            host_port = m.get("HostPort")
            break
    if not host_port and mappings:
        host_port = mappings[0].get("HostPort") if isinstance(mappings[0], dict) else None
    if not host_port:
        log.debug("No host port mapping yet for %s/tcp on instance %s", VAST_LLAMA_PORT, instance.get("id"))
        return None
    return f"{REMOTE_SCHEME}://{ip}:{host_port}"



async def probe_remote_llamacpp(base_url: str) -> bool:
    url = f"{base_url}{HEALTHCHECK_PATH}"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, headers={"Authorization": "Bearer dummy"})
            return r.status_code < 500
    except Exception:
        return False


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
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
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
    raise HTTPException(status_code=502, detail=f"Unexpected instance response: {list(payload.keys()) if isinstance(payload, dict) else payload}")


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
        raise HTTPException(status_code=502, detail="Offer missing ask ID")

    env: Dict[str, str] = {f"-p {VAST_LLAMA_PORT}:{VAST_LLAMA_PORT}": "1"}
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
        raise HTTPException(status_code=502, detail=f"Unexpected create-instance response: {payload}")
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
        raise HTTPException(status_code=503, detail="No matching RTX PRO 6000 / S / WS offers found on Vast")

    candidates.sort(key=rank_offer)
    best = candidates[0]
    best_price = extract_price_per_hour(best)
    if best_price is None:
        raise HTTPException(status_code=503, detail="Matching offer found but price could not be determined")

    if best_price > MAX_HOURLY_PRICE:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Cheapest matching RTX PRO 6000-family offer costs ${best_price:.2f}/hr, "
                f"above MAX_HOURLY_PRICE=${MAX_HOURLY_PRICE:.2f}/hr"
            ),
        )

    log.info(
        "Creating instance from cheapest offer id=%s ask_id=%s gpu=%s price=$%.2f/hr",
        best.get("id"),
        best.get("ask_contract_id"),
        best.get("gpu_name"),
        best_price,
    )
    return await vast_create_instance_from_offer(client, best)


async def ensure_instance_ready() -> str:
    global current_instance_id, current_remote_base_url, instance_ready

    async with state_lock:
        if instance_ready and current_instance_id and current_remote_base_url:
            return current_remote_base_url

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
                        if REMOTE_BASE_URL:
                            base = REMOTE_BASE_URL
                        else:
                            base = build_remote_base_direct(instance)
                            if not base:
                                log.info("Waiting for port mapping on instance %s...", instance_id)
                                await asyncio.sleep(HEALTHCHECK_INTERVAL)
                                continue
                        log.info("Probing llama.cpp health at %s", base)
                        if await probe_remote_llamacpp(base):
                            current_remote_base_url = base
                            instance_ready = True
                            log.info("Remote llama.cpp ready at %s for instance %s", base, instance_id)
                            return base
                        log.info("llama.cpp not yet healthy at %s, retrying...", base)
                except Exception as exc:
                    last_err = exc
                    log.warning("Error polling instance %s: %s", instance_id, exc)

                await asyncio.sleep(HEALTHCHECK_INTERVAL)

            detail = f"Timed out waiting for instance {instance_id} / llama.cpp readiness"
            if last_err:
                detail += f": {last_err}"
            raise HTTPException(status_code=504, detail=detail)


# -----------------------------------------------------------------------------
# Idle reaper
# -----------------------------------------------------------------------------

async def idle_reaper() -> None:
    global instance_ready, current_remote_base_url, current_instance_id

    while True:
        await asyncio.sleep(5)

        try:
            if not current_instance_id or not instance_ready or inflight_requests > 0:
                continue
            idle_for = now() - last_used_ts if last_used_ts else 0
            if idle_for < IDLE_SECONDS:
                continue

            async with state_lock:
                if not current_instance_id or not instance_ready or inflight_requests > 0:
                    continue
                idle_for = now() - last_used_ts if last_used_ts else 0
                if idle_for < IDLE_SECONDS:
                    continue

                async with httpx.AsyncClient(timeout=60.0) as client:
                    if INSTANCE_ACTION == "destroy":
                        log.info("Idle timeout: destroying instance %s", current_instance_id)
                        await vast_destroy_instance(client, current_instance_id)
                        current_instance_id = None
                    else:
                        log.info("Idle timeout: stopping instance %s", current_instance_id)
                        await vast_manage_instance(client, current_instance_id, "stopped")

                current_remote_base_url = None
                instance_ready = False

        except Exception:
            log.exception("Idle reaper error")


# -----------------------------------------------------------------------------
# Proxying
# -----------------------------------------------------------------------------

async def stream_upstream_response(
    method: str,
    url: str,
    headers: Dict[str, str],
    body: bytes,
) -> tuple[int, Dict[str, str], AsyncIterator[bytes]]:
    client = httpx.AsyncClient(timeout=httpx.Timeout(PROXY_TIMEOUT, connect=60.0))
    req = client.build_request(method, url, headers=headers, content=body)
    resp = await client.send(req, stream=True)

    async def iterator() -> AsyncIterator[bytes]:
        nonlocal client, resp
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    passthrough = {}
    for k, v in resp.headers.items():
        if k.lower() in {"content-type", "cache-control", "x-request-id", "openai-processing-ms"}:
            passthrough[k] = v

    return resp.status_code, passthrough, iterator()


async def proxy_request(request: Request, path: str) -> Response:
    global last_used_ts, inflight_requests

    base = await ensure_instance_ready()
    upstream_url = f"{base}/v1/{path}"
    body = await request.body()

    upstream_headers: Dict[str, str] = {}
    for k, v in request.headers.items():
        lk = k.lower()
        if lk in {"host", "content-length", "authorization"}:
            continue
        upstream_headers[k] = v

    upstream_headers["Authorization"] = "Bearer dummy"

    inflight_requests += 1
    last_used_ts = now()
    try:
        status_code, headers, iterator = await stream_upstream_response(
            method=request.method,
            url=upstream_url,
            headers=upstream_headers,
            body=body,
        )
        return StreamingResponse(
            iterator,
            status_code=status_code,
            headers=headers,
            media_type=headers.get("content-type"),
        )
    finally:
        inflight_requests -= 1
        last_used_ts = now()


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    global last_used_ts
    last_used_ts = now()
    asyncio.create_task(idle_reaper())
    log.info("Gateway started")

@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "service": "vast-roo-gateway",
        "openai_base": "/v1",
        "health": "/healthz",
    }

@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {
        "ok": True,
        "instance_id": current_instance_id,
        "instance_ready": instance_ready,
        "idle_seconds": IDLE_SECONDS,
        "max_hourly_price": MAX_HOURLY_PRICE,
        "instance_action": INSTANCE_ACTION,
    }

@app.get("/v1/models")
async def models(request: Request) -> Response:
    return await proxy_request(request, "models")

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def catchall_v1(path: str, request: Request) -> Response:
    return await proxy_request(request, path)

@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> PlainTextResponse:
    log.exception("Unhandled error: %s", exc)
    return PlainTextResponse("Internal gateway error", status_code=500)
