import asyncio
import os
import tempfile
from typing import List

from config import LOCAL_LLAMA_PORT, SERVER_SH, log


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
