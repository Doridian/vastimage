"""Tests for utils.py functions."""

import pytest
from utils import (
    check_tcp_port,
    extract_price_per_hour,
    extract_public_ip,
    gpu_matches,
    get_ssh_host_and_port,
    infer_running,
    instance_destroyed,
    now,
)


class TestNow:
    """Tests for the now() function."""

    def test_now_returns_float(self):
        """now() should return a float."""
        result = now()
        assert isinstance(result, float)

    def test_now_returns_positive_value(self):
        """now() should return a positive timestamp."""
        result = now()
        assert result > 0


class TestGpuMatches:
    """Tests for the gpu_matches() function."""

    def test_exact_match(self):
        """Should match exact GPU name."""
        assert gpu_matches("H100", "H100", []) is True

    def test_case_insensitive(self):
        """Should match regardless of case."""
        assert gpu_matches("H100", "h100", []) is True
        assert gpu_matches("h100", "H100", []) is True

    def test_with_underscores(self):
        """Should match with underscores (normalized to spaces)."""
        # H_100 -> H 100, H100 -> H100, these don't match
        assert gpu_matches("H_100", "H 100", []) is True

    def test_with_hyphens(self):
        """Should match with hyphens (normalized to spaces)."""
        # H-100 -> H 100, H100 -> H100, these don't match
        assert gpu_matches("H-100", "H 100", []) is True

    def test_partial_match(self):
        """Should match partial GPU name."""
        assert gpu_matches("NVIDIA H100 80GB", "H100", []) is True

    def test_exclude_match(self):
        """Should exclude matching tokens."""
        assert gpu_matches("H100 PCIE", "H100", ["PCIE"]) is False

    def test_exclude_multiple_tokens(self):
        """Should exclude with multiple exclude tokens."""
        assert gpu_matches("H100 PCIE 80GB", "H100", ["PCIE", "80GB"]) is False

    def test_no_exclude(self):
        """Should match when no exclude tokens provided."""
        assert gpu_matches("H100 PCIE", "H100", []) is True

    def test_no_match(self):
        """Should not match when GPU name doesn't contain search term."""
        assert gpu_matches("A100", "H100", []) is False


class TestExtractPricePerHour:
    """Tests for the extract_price_per_hour() function."""

    def test_dph_total(self):
        """Should extract from dph_total."""
        obj = {"dph_total": 1.50}
        assert extract_price_per_hour(obj) == 1.50

    def test_discounted_dph_total(self):
        """Should extract from discounted_dph_total."""
        obj = {"discounted_dph_total": 2.50}
        assert extract_price_per_hour(obj) == 2.50

    def test_dph_total_adj(self):
        """Should extract from dph_total_adj."""
        obj = {"dph_total_adj": 3.50}
        assert extract_price_per_hour(obj) == 3.50

    def test_discounted_hourly(self):
        """Should extract from discounted_hourly."""
        obj = {"discounted_hourly": 4.50}
        assert extract_price_per_hour(obj) == 4.50

    def test_priority_order(self):
        """Should prefer dph_total over discounted_dph_total."""
        obj = {"dph_total": 1.50, "discounted_dph_total": 2.50}
        assert extract_price_per_hour(obj) == 1.50

    def test_no_price_found(self):
        """Should return None when no price found."""
        obj = {"some_other_key": "value"}
        assert extract_price_per_hour(obj) is None

    def test_empty_dict(self):
        """Should return None for empty dict."""
        assert extract_price_per_hour({}) is None

    def test_non_numeric_value(self):
        """Should skip non-numeric values."""
        obj = {"dph_total": "not a number"}
        assert extract_price_per_hour(obj) is None


class TestExtractPublicIp:
    """Tests for the extract_public_ip() function."""

    def test_public_ipaddr(self):
        """Should extract from public_ipaddr."""
        obj = {"public_ipaddr": "192.168.1.1"}
        assert extract_public_ip(obj) == "192.168.1.1"

    def test_public_ip(self):
        """Should extract from public_ip."""
        obj = {"public_ip": "192.168.1.1"}
        assert extract_public_ip(obj) == "192.168.1.1"

    def test_ssh_host(self):
        """Should extract from ssh_host."""
        obj = {"ssh_host": "192.168.1.1"}
        assert extract_public_ip(obj) == "192.168.1.1"

    def test_host(self):
        """Should extract from host."""
        obj = {"host": "192.168.1.1"}
        assert extract_public_ip(obj) == "192.168.1.1"

    def test_actual_ip(self):
        """Should extract from actual_ip."""
        obj = {"actual_ip": "192.168.1.1"}
        assert extract_public_ip(obj) == "192.168.1.1"

    def test_priority_order(self):
        """Should prefer public_ipaddr over other keys."""
        obj = {"public_ipaddr": "192.168.1.1", "public_ip": "192.168.1.2"}
        assert extract_public_ip(obj) == "192.168.1.1"

    def test_no_ip_found(self):
        """Should return None when no IP found."""
        obj = {"some_other_key": "value"}
        assert extract_public_ip(obj) is None

    def test_empty_string(self):
        """Should skip empty strings."""
        obj = {"public_ipaddr": ""}
        assert extract_public_ip(obj) is None

    def test_whitespace_only(self):
        """Should skip whitespace-only strings."""
        obj = {"public_ipaddr": "   "}
        assert extract_public_ip(obj) is None


class TestInferRunning:
    """Tests for the infer_running() function."""

    def test_running_status(self):
        """Should detect running status."""
        obj = {"actual_status": "running"}
        assert infer_running(obj) is True

    def test_up_status(self):
        """Should detect 'up' status."""
        obj = {"cur_state": "up"}
        assert infer_running(obj) is True

    def test_running_in_value(self):
        """Should detect 'running' in value."""
        obj = {"status": "running"}
        assert infer_running(obj) is True

    def test_not_running(self):
        """Should not detect non-running status."""
        obj = {"actual_status": "stopped"}
        assert infer_running(obj) is False

    def test_no_status_key(self):
        """Should return False when no status key."""
        obj = {"some_other_key": "value"}
        assert infer_running(obj) is False

    def test_empty_dict(self):
        """Should return False for empty dict."""
        assert infer_running({}) is False


class TestInstanceDestroyed:
    """Tests for the instance_destroyed() function."""

    def test_destroyed_status(self):
        """Should detect destroyed status."""
        obj = {"actual_status": "destroyed"}
        assert instance_destroyed(obj) is True

    def test_destroying_status(self):
        """Should detect destroying status."""
        obj = {"status": "destroying"}
        assert instance_destroyed(obj) is True

    def test_destroyed_in_value(self):
        """Should detect 'destroy' in value."""
        obj = {"cur_state": "destroyed"}
        assert instance_destroyed(obj) is True

    def test_not_destroyed(self):
        """Should not detect non-destroyed status."""
        obj = {"actual_status": "running"}
        assert instance_destroyed(obj) is False

    def test_no_status_key(self):
        """Should return False when no status key."""
        obj = {"some_other_key": "value"}
        assert instance_destroyed(obj) is False

    def test_empty_dict(self):
        """Should return False for empty dict."""
        assert instance_destroyed({}) is False


class TestGetSshHostAndPort:
    """Tests for the get_ssh_host_and_port() function."""

    def test_basic_mapping(self):
        """Should extract host and port from basic mapping."""
        obj = {
            "public_ipaddr": "192.168.1.1",
            "ports": {"2222/tcp": [{"HostIp": "0.0.0.0", "HostPort": 2222}]}
        }
        result = get_ssh_host_and_port(obj)
        assert result == ("192.168.1.1", 2222)

    def test_no_host_ip(self):
        """Should handle missing HostIp."""
        obj = {
            "public_ipaddr": "192.168.1.1",
            "ports": {"2222/tcp": [{"HostPort": 2222}]}
        }
        result = get_ssh_host_and_port(obj)
        assert result == ("192.168.1.1", 2222)

    def test_empty_host_ip(self):
        """Should handle empty HostIp."""
        obj = {
            "public_ipaddr": "192.168.1.1",
            "ports": {"2222/tcp": [{"HostIp": "", "HostPort": 2222}]}
        }
        result = get_ssh_host_and_port(obj)
        assert result == ("192.168.1.1", 2222)

    def test_first_mapping_fallback(self):
        """Should use first mapping when no HostIp."""
        obj = {
            "public_ipaddr": "192.168.1.1",
            "ports": {"2222/tcp": [{"HostPort": 2222}]}
        }
        result = get_ssh_host_and_port(obj)
        assert result == ("192.168.1.1", 2222)

    def test_no_mapping(self):
        """Should return None when no port mapping."""
        obj = {
            "public_ipaddr": "192.168.1.1",
            "ports": {}
        }
        result = get_ssh_host_and_port(obj)
        assert result is None

    def test_no_public_ip(self):
        """Should return None when no public IP."""
        obj = {
            "ports": {"2222/tcp": [{"HostIp": "0.0.0.0", "HostPort": 2222}]}
        }
        result = get_ssh_host_and_port(obj)
        assert result is None

    def test_empty_dict(self):
        """Should return None for empty dict."""
        assert get_ssh_host_and_port({}) is None


class TestCheckTcpPort:
    """Tests for the check_tcp_port() function."""

    @pytest.mark.asyncio
    async def test_open_port(self):
        """Should return True for open port."""
        import asyncio
        # Start a simple server
        server = await asyncio.start_server(
            lambda reader, writer: None,
            '127.0.0.1', 0
        )
        server_port = server.sockets[0].getsockname()[1]
        
        try:
            result = await check_tcp_port('127.0.0.1', server_port)
            assert result is True
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_closed_port(self):
        """Should return False for closed port."""
        result = await check_tcp_port('127.0.0.1', 59999)
        assert result is False
