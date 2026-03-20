import asyncio
from typing import Any, Dict, List, Optional

import httpx

from config import (
    INSTANCE_LABEL,
    PREFER_VERIFIED,
    REQUIRE_RELIABILITY_GTE,
    SEARCH_LIMIT,
    VAST_API_BASE,
    VAST_API_KEY,
    VAST_DISK_GB,
    VAST_IMAGE,
)


def vast_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {VAST_API_KEY}",
        "Content-Type": "application/json",
    }


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
