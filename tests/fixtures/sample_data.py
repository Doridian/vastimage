"""Sample data fixtures for testing."""

# Sample offer data
SAMPLE_OFFER = {
    "id": 12345,
    "gpu_name": "H100",
    "dph_total": 1.50,
    "reliability": 0.98,
    "verified": True,
    "location": "US-East",
    "gpu_ram": 81920,
    "num_gpus": 1,
    "rentable": True,
    "rented": False,
    "gpu_arch": "nvidia",
}

SAMPLE_OFFER_LIST = [
    SAMPLE_OFFER,
    {
        "id": 12346,
        "gpu_name": "H200",
        "dph_total": 2.50,
        "reliability": 0.95,
        "verified": True,
        "location": "US-West",
        "gpu_ram": 147456,
        "num_gpus": 1,
        "rentable": True,
        "rented": False,
        "gpu_arch": "nvidia",
    },
    {
        "id": 12347,
        "gpu_name": "H100 PCIE",
        "dph_total": 1.25,
        "reliability": 0.92,
        "verified": False,
        "location": "EU",
        "gpu_ram": 81920,
        "num_gpus": 1,
        "rentable": True,
        "rented": False,
        "gpu_arch": "nvidia",
    },
]

# Sample instance data
SAMPLE_INSTANCE = {
    "id": 54321,
    "label": "vastimage-controlled",
    "actual_status": "running",
    "public_ipaddr": "192.168.1.100",
    "ports": {
        "2222/tcp": [
            {"HostIp": "0.0.0.0", "HostPort": 2222}
        ]
    },
    "dph_total": 1.50,
    "reliability": 0.98,
}

SAMPLE_INSTANCE_LIST = [
    SAMPLE_INSTANCE,
    {
        "id": 54322,
        "label": "vastimage-controlled",
        "actual_status": "stopped",
        "public_ipaddr": "192.168.1.101",
        "ports": {},
        "dph_total": 1.25,
        "reliability": 0.95,
    },
]

# Sample SSH host public keys
SAMPLE_HOST_KEYS = [
    "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC7test123",
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAItest456",
]

# Sample script content
SAMPLE_SCRIPT = """#!/bin/bash
echo "Starting server..."
python -m http.server 8080
"""
