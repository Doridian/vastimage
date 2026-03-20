import logging
import os

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

VAST_API_KEY = os.environ.get("VAST_API_KEY", "").strip()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("vast-instance")
