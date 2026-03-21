import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

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
from vast_api import VastAPI

log = logging.getLogger("vast-instance")


class Instance:
    def __init__(
        self,
        api: VastAPI,
        *,
        gpu_search: str,
        gpu_exclude: List[str],
        script: str,
        max_hourly_price: float = 5.0,
        startup_timeout: int = 3600,
        healthcheck_interval: float = 3.0,
        instance_action: str = "stop",
        vast_image: str = "ghcr.io/doridian/vastimage/vastimage:latest",
        vast_disk_gb: float = 100.0,
        instance_label: str = "vastimage-controlled",
        local_port: int = 6969,
        prefer_verified: bool = True,
        require_reliability_gte: float = 0.95,
        search_limit: int = 500,
        offer_selector: Optional[Callable[[List[Dict[str, Any]]], Awaitable[Dict[str, Any]]]] = None,
        existing_instance_selector: Optional[Callable[[List[Dict[str, Any]]], Awaitable[Optional[Dict[str, Any]]]]] = None,
    ) -> None:
        self._api = api
        self._gpu_search = gpu_search
        self._gpu_exclude = gpu_exclude
        self._script = script
        self._max_hourly_price = max_hourly_price
        self._startup_timeout = startup_timeout
        self._healthcheck_interval = healthcheck_interval
        self._instance_action = instance_action
        self._vast_image = vast_image
        self._vast_disk_gb = vast_disk_gb
        self._instance_label = instance_label
        self._local_port = local_port
        self._prefer_verified = prefer_verified
        self._require_reliability_gte = require_reliability_gte
        self._search_limit = search_limit
        self._offer_selector = offer_selector
        self._existing_instance_selector = existing_instance_selector

        self._instance_id: Optional[int] = None
        self._ssh_process: Optional[asyncio.subprocess.Process] = None
        self._known_hosts_file: Optional[str] = None

    @asynccontextmanager
    async def start(self) -> AsyncIterator["Instance"]:
        await self._ensure_ready()
        try:
            yield self
        finally:
            await self._cleanup()

    async def wait(self) -> None:
        """Wait until the SSH tunnel exits"""
        if self._ssh_process is not None:
            await self._ssh_process.wait()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rank_existing(self, instance: Dict[str, Any]) -> Tuple[int, float, int]:
        running_score = 0 if infer_running(instance) else 1
        price = extract_price_per_hour(instance) or 999999.0
        return (running_score, price, int(instance.get("id", 1_000_000_000)))

    @staticmethod
    def _rank_offer(offer: Dict[str, Any]) -> Tuple[float, float, int]:
        price = extract_price_per_hour(offer) or 999999.0
        reliability = float(offer.get("reliability", 0.0) or 0.0)
        return (price, -reliability, int(offer.get("id", 1_000_000_000)))

    async def _choose_or_create(
        self,
        client: httpx.AsyncClient,
        offers: List[Dict[str, Any]],
    ) -> int:
        instances = await self._api.show_instances(client)
        all_active = [i for i in instances if not instance_destroyed(i)]

        if self._existing_instance_selector is not None and all_active:
            result = await self._existing_instance_selector(all_active)
            if result is not None:
                chosen = result
                iid = int(chosen["id"])
                log.info(
                    "Reusing existing instance id=%s gpu=%s status=%s",
                    iid, chosen.get("gpu_name"),
                    chosen.get("actual_status") or chosen.get("cur_state") or chosen.get("status"),
                )
                return iid
        else:
            matches = [
                i for i in all_active
                if gpu_matches(str(i.get("gpu_name") or ""), self._gpu_search, self._gpu_exclude)
                and i.get("label") == self._instance_label
            ]
            if matches:
                matches.sort(key=self._rank_existing)
                chosen = matches[0]
                iid = int(chosen["id"])
                log.info(
                    "Reusing existing instance id=%s gpu=%s status=%s",
                    iid, chosen.get("gpu_name"),
                    chosen.get("actual_status") or chosen.get("cur_state") or chosen.get("status"),
                )
                return iid

        candidates = [
            o for o in offers
            if gpu_matches(str(o.get("gpu_name") or ""), self._gpu_search, self._gpu_exclude)
        ]
        if not candidates:
            raise RuntimeError(f"No matching '{self._gpu_search}' offers found on Vast")

        candidates.sort(key=self._rank_offer)

        if self._offer_selector is not None:
            best = await self._offer_selector(candidates)
        else:
            best = candidates[0]

        best_price = extract_price_per_hour(best)
        if best_price is None:
            raise RuntimeError("Matching offer found but price could not be determined")
        if best_price > self._max_hourly_price:
            raise RuntimeError(
                f"Cheapest '{self._gpu_search}' offer costs ${best_price:.2f}/hr, "
                f"above --max-hourly-price=${self._max_hourly_price:.2f}/hr"
            )

        log.info(
            "Creating instance from offer id=%s gpu=%s price=$%.2f/hr",
            best.get("id"), best.get("gpu_name"), best_price,
        )
        return await self._api.create_instance_from_offer(
            client, best,
            disk_gb=self._vast_disk_gb,
            label=self._instance_label,
            image=self._vast_image,
        )

    async def _ensure_ready(self) -> None:
        async with httpx.AsyncClient(timeout=60.0) as client:
            log.info("Fetching Vast.ai GPU offers")
            offers = await self._api.search_offers(
                client,
                prefer_verified=self._prefer_verified,
                require_reliability_gte=self._require_reliability_gte,
                search_limit=self._search_limit,
            )

            instance_id = await self._choose_or_create(client, offers)
            self._instance_id = instance_id

            instance = await self._api.show_instance(client, instance_id)
            if not infer_running(instance):
                log.info("Starting instance %s", instance_id)
                await self._api.manage_instance(client, instance_id, "running")

            deadline = now() + self._startup_timeout
            last_err: Optional[Exception] = None
            ip: Optional[str] = None
            host_port: Optional[int] = None

            while now() < deadline:
                try:
                    instance = await self._api.show_instance(client, instance_id)
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
                            await asyncio.sleep(self._healthcheck_interval)
                            continue
                        ip, host_port = conn
                        if not await check_tcp_port(ip, host_port):
                            log.info("SSH port not yet reachable at %s:%s, retrying...", ip, host_port)
                            ip = None
                            host_port = None
                            await asyncio.sleep(self._healthcheck_interval)
                            continue
                        break
                except Exception as exc:
                    last_err = exc
                    log.warning("Error polling instance %s: %s", instance_id, exc)

                await asyncio.sleep(self._healthcheck_interval)

            if not ip or not host_port:
                detail = f"Timed out waiting for instance {instance_id} / SSH readiness"
                if last_err:
                    detail += f": {last_err}"
                raise RuntimeError(detail)

            log.info("Fetching host public keys for instance %s", instance_id)
            keys = await self._api.fetch_host_pubkeys(client, instance_id)
            if not keys:
                if keys is None:
                    raise RuntimeError(
                        f"Failed to fetch host public keys for instance {instance_id}: Not in log"
                    )
                else:
                    raise RuntimeError(
                        f"Failed to fetch host public keys for instance {instance_id}: Empty keys block"
                    )
            self._known_hosts_file = write_known_hosts(ip, host_port, keys)
            log.info("Host keys written to %s", self._known_hosts_file)
            self._ssh_process = await connect_instance(
                ip, host_port, self._known_hosts_file,
                self._script, self._local_port,
            )
            log.info("Connected to instance %s", instance_id)

    async def _cleanup(self) -> None:
        if self._ssh_process is not None:
            log.info("Terminating SSH tunnel")
            self._ssh_process.terminate()
            try:
                await asyncio.wait_for(self._ssh_process.wait(), timeout=5.0)
            except Exception:
                self._ssh_process.kill()

        if self._known_hosts_file is not None:
            os.unlink(self._known_hosts_file)

        if self._instance_id is not None:
            async with httpx.AsyncClient(timeout=60.0) as client:
                if self._instance_action == "destroy":
                    log.info("Destroying instance %s", self._instance_id)
                    await self._api.destroy_instance(client, self._instance_id)
                else:
                    log.info("Stopping instance %s", self._instance_id)
                    await self._api.manage_instance(client, self._instance_id, "stopped")
