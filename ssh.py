import asyncio
import os
import tempfile
from typing import List


def write_known_hosts(ip: str, host_port: int, keys: List[str]) -> str:
    fd, path = tempfile.mkstemp(suffix=".known_hosts")
    with os.fdopen(fd, "w") as f:
        for key in keys:
            f.write(f"[{ip}]:{host_port} {key}\n")
    return path


async def connect_instance(
    ip: str,
    host_port: int,
    known_hosts_path: str,
    model_name: str,
    script_template: str,
    local_port: int,
) -> asyncio.subprocess.Process:
    script = script_template.format(model_name=model_name)
    cmd = [
        "ssh",
        "-L", f"{local_port}:127.0.0.1:8080",
        "-p", str(host_port),
        "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={known_hosts_path}",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3",
        f"fox@{ip}",
        "sh",
    ]
    import logging
    logging.getLogger("vast-instance").info("Connecting to instance: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.PIPE)
    proc.stdin.write(script.encode())
    await proc.stdin.drain()
    proc.stdin.close()
    return proc
