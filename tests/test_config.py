import pytest
import os
from src.config import Config

def test_config_validation_fails_without_keys(monkeypatch):
    monkeypatch.setattr(Config, "OPENAI_API_KEY", None)
    monkeypatch.setattr(Config, "PINECONE_API_KEY", None)
    with pytest.raises(ValueError):
        Config.validate()

def test_config_validation_passes_with_keys(monkeypatch):
    monkeypatch.setattr(Config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(Config, "PINECONE_API_KEY", "test-key")
    # Should not raise exception
    Config.validate()
