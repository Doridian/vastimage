import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import (
    HEALTHCHECK_INTERVAL,
    MAX_HOURLY_PRICE,
    MODEL_NAME,
    PICK_CONFIGS,
    STARTUP_TIMEOUT,
    log,
)
from ssh import connect_instance, write_known_hosts
from utils import (
    check_tcp_port,
    extract_price_per_hour,
    get_ssh_host_and_port,
    gpu_matches,
    infer_running,
    instance_destroyed,
    now,
)
from vast_api import (
    fetch_host_pubkeys,
    vast_create_instance_from_offer,
    vast_manage_instance,
    vast_search_offers,
    vast_show_instance,
    vast_show_instances,
)

# -----------------------------------------------------------------------------
# Mutable state
# -----------------------------------------------------------------------------

current_instance_id: Optional[int] = None
ssh_process: Optional[asyncio.subprocess.Process] = None
known_hosts_file: Optional[str] = None

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
