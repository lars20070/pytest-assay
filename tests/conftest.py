#!/usr/bin/env python3

from __future__ import annotations

import httpx
import pytest
from _ollama import OLLAMA_BASE_URL, OLLAMA_MODEL
from pydantic_ai.settings import ModelSettings


def _ollama_is_running() -> bool:
    """Check whether the local Ollama server is reachable."""
    try:
        response = httpx.get(OLLAMA_BASE_URL, timeout=5)
        return response.status_code == 200
    except httpx.ConnectError:
        return False


def _ollama_model_available() -> bool:
    """Check whether OLLAMA_MODEL is available on the local Ollama server."""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            return False
        models = response.json().get("models", [])
        return any(m.get("name") == OLLAMA_MODEL for m in models)
    except (httpx.ConnectError, httpx.HTTPError):
        return False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "ollama: tests requiring a local Ollama instance")


@pytest.fixture(autouse=True)
def skip_ollama_tests(request: pytest.FixtureRequest) -> None:
    """Skip tests marked with 'ollama' when running in CI or when Ollama is not reachable."""
    if not request.node.get_closest_marker("ollama"):
        return
    if not _ollama_is_running():
        pytest.skip(f"Ollama server is not running at {OLLAMA_BASE_URL}. Please start it with `ollama serve` before running this test.")
    if not _ollama_model_available():
        pytest.skip(f"Ollama model '{OLLAMA_MODEL}' is not available. Please download it with `ollama pull {OLLAMA_MODEL}` before running this test.")


@pytest.fixture
def ollama_base_url() -> str:
    """Provide the Ollama base URL for unit test assertions."""
    return OLLAMA_BASE_URL


@pytest.fixture
def ollama_model() -> str:
    """Provide the Ollama model name for unit test assertions."""
    return OLLAMA_MODEL


@pytest.fixture
def model_settings() -> ModelSettings:
    """Provide deterministic model settings for VCR-compatible tests."""
    return ModelSettings(temperature=0.0, timeout=300)
