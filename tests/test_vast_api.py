"""Tests for vast_api.py with mocking."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from vast_api import VastAPI


class TestVastAPI:
    """Tests for the VastAPI class."""

    def test_init(self):
        """Should initialize with API key and base URL."""
        api = VastAPI("test-key", "https://example.com/api")
        assert api._api_key == "test-key"
        assert api._api_base == "https://example.com/api"

    def test_init_strips_trailing_slash(self):
        """Should strip trailing slash from base URL."""
        api = VastAPI("test-key", "https://example.com/api/")
        assert api._api_base == "https://example.com/api"

    def test_headers(self):
        """Should return correct headers."""
        api = VastAPI("test-key")
        headers = api._headers()
        assert headers == {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        }


class TestShowInstances:
    """Tests for show_instances method."""

    @pytest.mark.asyncio
    async def test_show_instances_success(self):
        """Should return instances from API response."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"instances": [{"id": 1}, {"id": 2}]}
        
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        
        result = await api.show_instances(mock_client)
        
        assert result == [{"id": 1}, {"id": 2}]
        mock_client.get.assert_called_once_with(
            "https://console.vast.ai/api/v0/instances/",
            headers={"Authorization": "Bearer test-key", "Content-Type": "application/json"},
        )

    @pytest.mark.asyncio
    async def test_show_instances_empty(self):
        """Should return empty list when no instances."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {}
        
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        
        result = await api.show_instances(mock_client)
        
        assert result == []


class TestShowInstance:
    """Tests for show_instance method."""

    @pytest.mark.asyncio
    async def test_show_instance_success(self):
        """Should return instance from API response."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"instances": {"id": 1}}
        
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        
        result = await api.show_instance(mock_client, 1)
        
        assert result == {"id": 1}

    @pytest.mark.asyncio
    async def test_show_instance_list_response(self):
        """Should return first instance from list response."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"instances": [{"id": 1}, {"id": 2}]}
        
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        
        result = await api.show_instance(mock_client, 1)
        
        assert result == {"id": 1}

    @pytest.mark.asyncio
    async def test_show_instance_not_found(self):
        """Should raise RuntimeError for 404."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        
        with pytest.raises(RuntimeError, match="Instance 1 not found"):
            await api.show_instance(mock_client, 1)

    @pytest.mark.asyncio
    async def test_show_instance_unexpected_response(self):
        """Should raise RuntimeError for unexpected response."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"unexpected": "data"}
        
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        
        with pytest.raises(RuntimeError, match="Unexpected instance response"):
            await api.show_instance(mock_client, 1)


class TestManageInstance:
    """Tests for manage_instance method."""

    @pytest.mark.asyncio
    async def test_manage_instance_success(self):
        """Should send correct request to manage instance."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.put.return_value = mock_response
        
        await api.manage_instance(mock_client, 1, "running")
        
        mock_client.put.assert_called_once_with(
            "https://console.vast.ai/api/v0/instances/1/",
            headers={"Authorization": "Bearer test-key", "Content-Type": "application/json"},
            json={"state": "running"},
        )


class TestDestroyInstance:
    """Tests for destroy_instance method."""

    @pytest.mark.asyncio
    async def test_destroy_instance_success(self):
        """Should send correct request to destroy instance."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.delete.return_value = mock_response
        
        await api.destroy_instance(mock_client, 1)
        
        mock_client.delete.assert_called_once_with(
            "https://console.vast.ai/api/v0/instances/1/",
            headers={"Authorization": "Bearer test-key", "Content-Type": "application/json"},
        )


class TestSearchOffers:
    """Tests for search_offers method."""

    @pytest.mark.asyncio
    async def test_search_offers_success(self):
        """Should return offers from API response."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "offers": [{"id": 1, "dph_total": 1.0}, {"id": 2, "dph_total": 2.0}]
        }
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        result = await api.search_offers(
            mock_client,
            prefer_verified=True,
            require_reliability_gte=0.95,
            search_limit=500,
        )
        
        assert result == [{"id": 1, "dph_total": 1.0}, {"id": 2, "dph_total": 2.0}]

    @pytest.mark.asyncio
    async def test_search_offers_dict_response(self):
        """Should handle single offer dict response."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"offers": {"id": 1}}
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        result = await api.search_offers(
            mock_client,
            prefer_verified=True,
            require_reliability_gte=0.95,
            search_limit=500,
        )
        
        assert result == [{"id": 1}]

    @pytest.mark.asyncio
    async def test_search_offers_nested_offers(self):
        """Should extract from nested offers structure."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "offers": [{"offers": {"id": 1}}, {"offers": {"id": 2}}]
        }
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        result = await api.search_offers(
            mock_client,
            prefer_verified=True,
            require_reliability_gte=0.95,
            search_limit=500,
        )
        
        assert result == [{"id": 1}, {"id": 2}]


class TestCreateInstanceFromOffer:
    """Tests for create_instance_from_offer method."""

    @pytest.mark.asyncio
    async def test_create_instance_success(self):
        """Should return new contract ID."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"new_contract": 123}
        
        mock_client = AsyncMock()
        mock_client.put.return_value = mock_response
        
        offer = {"ask_contract_id": 456, "id": 456}
        
        result = await api.create_instance_from_offer(
            mock_client,
            offer,
            disk_gb=100.0,
            label="test-label",
            image="test-image",
        )
        
        assert result == 123

    @pytest.mark.asyncio
    async def test_create_instance_missing_ask_id(self):
        """Should raise RuntimeError when offer missing ask ID."""
        api = VastAPI("test-key")
        
        mock_client = AsyncMock()
        
        offer = {}  # Missing both ask_contract_id and id
        
        with pytest.raises(RuntimeError, match="Offer missing ask ID"):
            await api.create_instance_from_offer(
                mock_client,
                offer,
                disk_gb=100.0,
                label="test-label",
                image="test-image",
            )

    @pytest.mark.asyncio
    async def test_create_instance_unexpected_response(self):
        """Should raise RuntimeError for unexpected response."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": "data"}
        
        mock_client = AsyncMock()
        mock_client.put.return_value = mock_response
        
        offer = {"ask_contract_id": 456}
        
        with pytest.raises(RuntimeError, match="Unexpected create-instance response"):
            await api.create_instance_from_offer(
                mock_client,
                offer,
                disk_gb=100.0,
                label="test-label",
                image="test-image",
            )


class TestFetchHostPubkeys:
    """Tests for fetch_host_pubkeys method."""

    @pytest.mark.asyncio
    async def test_fetch_host_pubkeys_success(self):
        """Should return host public keys."""
        api = VastAPI("test-key")
        
        # Mock the first response
        mock_response1 = MagicMock()
        mock_response1.is_success = True
        mock_response1.json.return_value = {"result_url": "/logs/123"}
        
        # Mock the second response
        # The code reverses lines, so BEGIN should come before END in the original text
        # After reversing: END comes first, then keys, then BEGIN
        mock_response2 = MagicMock()
        mock_response2.is_success = True
        mock_response2.text = "\n".join([
            "more log lines",
            "===BEGIN HOST PUBLIC KEYS===",
            "ssh-rsa AAAA...key1",
            "ssh-rsa AAAA...key2",
            "===END HOST PUBLIC KEYS===",
            "some log line",
        ])
        
        mock_client = AsyncMock()
        mock_client.put.return_value = mock_response1
        mock_client.get.return_value = mock_response2
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await api.fetch_host_pubkeys(mock_client, 1)
        
        assert result == ["ssh-rsa AAAA...key2", "ssh-rsa AAAA...key1"]

    @pytest.mark.asyncio
    async def test_fetch_host_pubkeys_failure(self):
        """Should return None on failure."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.is_success = False
        
        mock_client = AsyncMock()
        mock_client.put.return_value = mock_response
        
        result = await api.fetch_host_pubkeys(mock_client, 1)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_host_pubkeys_no_result_url(self):
        """Should return None when no result URL."""
        api = VastAPI("test-key")
        
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {}
        
        mock_client = AsyncMock()
        mock_client.put.return_value = mock_response
        
        result = await api.fetch_host_pubkeys(mock_client, 1)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_host_pubkeys_no_keys(self):
        """Should return None when no keys found."""
        api = VastAPI("test-key")
        
        mock_response1 = MagicMock()
        mock_response1.is_success = True
        mock_response1.json.return_value = {"result_url": "/logs/123"}
        
        mock_response2 = MagicMock()
        mock_response2.is_success = True
        mock_response2.text = "some log line\nmore log lines"
        
        mock_client = AsyncMock()
        mock_client.put.return_value = mock_response1
        mock_client.get.return_value = mock_response2
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await api.fetch_host_pubkeys(mock_client, 1)
        
        assert result is None
