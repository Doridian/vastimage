import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("vast-instance")


def now() -> float:
    return time.time()


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
