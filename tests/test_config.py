"""Tests for config.py."""

import os
from unittest.mock import patch

import pytest

# Need to reload config to pick up environment changes
import importlib
import config


class TestConfig:
    """Tests for config module."""

    def test_vast_api_key_empty_when_not_set(self):
        """VAST_API_KEY should be empty when not set."""
        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(config)
            assert config.VAST_API_KEY == ""

    def test_vast_api_key_when_set(self):
        """VAST_API_KEY should be set from environment."""
        with patch.dict(os.environ, {"VAST_API_KEY": "test-key-123"}):
            importlib.reload(config)
            assert config.VAST_API_KEY == "test-key-123"

    def test_vast_api_key_strips_whitespace(self):
        """VAST_API_KEY should strip whitespace."""
        with patch.dict(os.environ, {"VAST_API_KEY": "  test-key-123  "}):
            importlib.reload(config)
            assert config.VAST_API_KEY == "test-key-123"

    def test_log_exists(self):
        """log should exist in config."""
        assert hasattr(config, 'log')
        assert config.log is not None
