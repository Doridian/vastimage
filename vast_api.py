from typing import Any, Dict, List, Optional

import httpx


class VastAPI:
    def __init__(
        self,
        api_key: str,
        api_base: str = "https://console.vast.ai/api/v0",
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def show_instances(self, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
        r = await client.get(f"{self._api_base}/instances/", headers=self._headers())
        r.raise_for_status()
        return r.json().get("instances", [])

    async def show_instance(self, client: httpx.AsyncClient, instance_id: int) -> Dict[str, Any]:
        r = await client.get(f"{self._api_base}/instances/{instance_id}/", headers=self._headers())
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
        raise RuntimeError(
            f"Unexpected instance response: {list(payload.keys()) if isinstance(payload, dict) else payload}"
        )

    async def manage_instance(self, client: httpx.AsyncClient, instance_id: int, new_state: str) -> None:
        r = await client.put(
            f"{self._api_base}/instances/{instance_id}/",
            headers=self._headers(),
            json={"state": new_state},
        )
        r.raise_for_status()

    async def destroy_instance(self, client: httpx.AsyncClient, instance_id: int) -> None:
        r = await client.delete(f"{self._api_base}/instances/{instance_id}/", headers=self._headers())
        r.raise_for_status()

    async def search_offers(
        self,
        client: httpx.AsyncClient,
        *,
        prefer_verified: bool,
        require_reliability_gte: float,
        search_limit: int,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {
            "limit": search_limit,
            "type": "on-demand",
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "gpu_arch": {"eq": "nvidia"},
            "num_gpus": {"eq": 1},
            "gpu_ram": {"gte": 70000},
            "reliability": {"gte": require_reliability_gte},
            "order": [["dph_total", "asc"]],
            "disable_bundling": True,
        }
        if prefer_verified:
            body["verified"] = {"eq": True}

        r = await client.post(f"{self._api_base}/bundles", headers=self._headers(), json=body)
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

    async def create_instance_from_offer(
        self,
        client: httpx.AsyncClient,
        offer: Dict[str, Any],
        *,
        disk_gb: float,
        label: str,
        image: str,
    ) -> int:
        ask_id = offer.get("ask_contract_id") or offer.get("id")
        if not ask_id:
            raise RuntimeError("Offer missing ask ID")

        r = await client.put(
            f"{self._api_base}/asks/{ask_id}/",
            headers=self._headers(),
            json={
                "disk": disk_gb,
                "target_state": "running",
                "label": label,
                "cancel_unavail": True,
                "runtype": "args",
                "image": image,
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

    async def fetch_host_pubkeys(
        self, client: httpx.AsyncClient, instance_id: int
    ) -> Optional[List[str]]:
        r = await client.put(
            f"{self._api_base}/instances/request_logs/{instance_id}/",
            headers=self._headers(),
            json={"tail": "100"},
        )
        if not r.is_success:
            return None
        result_url = r.json().get("result_url")
        if not result_url:
            return None
        import asyncio
        await asyncio.sleep(3.0)
        r2 = await client.get(result_url)
        if not r2.is_success:
            return None
        lines = [line.strip() for line in r2.text.splitlines()]
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
