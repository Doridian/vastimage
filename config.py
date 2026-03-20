import logging
import os

# -----------------------------------------------------------------------------
# Remote server script (sent over SSH stdin on connect)
# -----------------------------------------------------------------------------

SERVER_SH = """\
#!/bin/sh
set -e

exec trtllm-serve "{model_name}" \\
    --host 127.0.0.1 \\
    --port 8080 \\
    {quant_flag}
"""

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

VAST_API_KEY = os.environ.get("VAST_API_KEY", "").strip()
VAST_API_BASE = os.environ.get("VAST_API_BASE", "https://console.vast.ai/api/v0").rstrip("/")

LOCAL_LLAMA_PORT = 6969

STARTUP_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT", "3600"))
HEALTHCHECK_INTERVAL = float(os.environ.get("HEALTHCHECK_INTERVAL", "3"))

MAX_HOURLY_PRICE = float(os.environ.get("MAX_HOURLY_PRICE", "1.00"))
PREFER_VERIFIED = os.environ.get("PREFER_VERIFIED", "true").lower() == "true"
REQUIRE_RELIABILITY_GTE = float(os.environ.get("REQUIRE_RELIABILITY_GTE", "0.95"))
SEARCH_LIMIT = int(os.environ.get("SEARCH_LIMIT", "500"))

VAST_IMAGE = os.environ.get("VAST_IMAGE", "ghcr.io/doridian/vastimage/vastimage:latest").strip()

VAST_DISK_GB = float(os.environ.get("VAST_DISK_GB", "100"))
INSTANCE_LABEL = os.environ.get("INSTANCE_LABEL", "vastimage-controlled").strip()

MODEL_NAME = os.environ.get("MODEL_NAME", "").strip()

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
# Pick configs: fixed GPU per quantization mode
# -----------------------------------------------------------------------------

PICK_CONFIGS = {
    "int4-awq": {"gpu_search": "H100", "gpu_exclude": ["PCIE"], "trtllm_quant": "w4a16_awq"},
    "fp8":      {"gpu_search": "H200", "gpu_exclude": [],       "trtllm_quant": "fp8"},
}
