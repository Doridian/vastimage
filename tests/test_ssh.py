"""Tests for ssh.py functions."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ssh import connect_instance, write_known_hosts


class TestWriteKnownHosts:
    """Tests for the write_known_hosts() function."""

    def test_writes_keys(self):
        """Should write keys to file."""
        keys = ["key1", "key2", "key3"]
        path = write_known_hosts("192.168.1.1", 2222, keys)
        
        try:
            with open(path, "r") as f:
                content = f.read()
            
            assert "[192.168.1.1]:2222 key1\n" in content
            assert "[192.168.1.1]:2222 key2\n" in content
            assert "[192.168.1.1]:2222 key3\n" in content
        finally:
            os.unlink(path)

    def test_creates_temp_file(self):
        """Should create a temporary file."""
        keys = ["key1"]
        path = write_known_hosts("192.168.1.1", 2222, keys)
        
        assert os.path.exists(path)
        assert path.endswith(".known_hosts")

    def test_file_is_deleted_after(self):
        """File should be deletable after use."""
        keys = ["key1"]
        path = write_known_hosts("192.168.1.1", 2222, keys)
        
        assert os.path.exists(path)
        os.unlink(path)
        assert not os.path.exists(path)


class TestConnectInstance:
    """Tests for the connect_instance() function."""

    @pytest.mark.asyncio
    async def test_connect_instance_creates_process(self):
        """Should create subprocess with correct arguments."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.known_hosts', delete=False) as f:
            f.write("key1\n")
            known_hosts_path = f.name
        
        try:
            script = "echo test"
            
            with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
                mock_proc = MagicMock()
                mock_proc.stdin = MagicMock()
                mock_proc.stdin.write = MagicMock()
                mock_proc.stdin.drain = AsyncMock()
                mock_proc.stdin.close = MagicMock()
                mock_exec.return_value = mock_proc
                
                result = await connect_instance(
                    "192.168.1.1",
                    2222,
                    known_hosts_path,
                    script,
                    6969,
                )
                
                assert result == mock_proc
                
                # Verify the command
                mock_exec.assert_called_once()
                call_args = mock_exec.call_args[0]
                assert "ssh" in call_args
                assert "-L" in call_args
                assert "6969:127.0.0.1:8080" in call_args
                assert "-p" in call_args
                assert "2222" in call_args
                assert "fox@192.168.1.1" in call_args
                
                # Verify stdin was written
                mock_proc.stdin.write.assert_called_once_with(script.encode())
                mock_proc.stdin.drain.assert_awaited_once()
                mock_proc.stdin.close.assert_called_once()
        finally:
            os.unlink(known_hosts_path)

    @pytest.mark.asyncio
    async def test_connect_instance_with_different_port(self):
        """Should use different local port."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.known_hosts', delete=False) as f:
            f.write("key1\n")
            known_hosts_path = f.name
        
        try:
            script = "echo test"
            
            with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
                mock_proc = MagicMock()
                mock_proc.stdin = MagicMock()
                mock_proc.stdin.write = MagicMock()
                mock_proc.stdin.drain = AsyncMock()
                mock_proc.stdin.close = MagicMock()
                mock_exec.return_value = mock_proc
                
                result = await connect_instance(
                    "192.168.1.1",
                    2222,
                    known_hosts_path,
                    script,
                    8080,
                )
                
                call_args = mock_exec.call_args[0]
                assert "8080:127.0.0.1:8080" in call_args
        finally:
            os.unlink(known_hosts_path)
