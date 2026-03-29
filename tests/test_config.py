#!/usr/bin/env python3

from unittest.mock import patch

from pytest_assay.config import Config, config


# @lat: [[tests/config-tests#Config Tests#Default Values]]
def test_config_default_values() -> None:
    """Test that Config has correct default values."""
    with patch.dict("os.environ", {}, clear=True):
        c = Config(
            _env_file=None,  # type: ignore[call-arg]
        )
    assert c.ollama_base_url == "http://localhost:11434"
    assert c.ollama_model == "qwen2.5:14b"


# @lat: [[tests/config-tests#Config Tests#Environment Variable Override]]
def test_config_env_override() -> None:
    """Test that Config reads values from environment variables."""
    with patch.dict(
        "os.environ",
        {
            "OLLAMA_BASE_URL": "http://custom-host:9999",
            "OLLAMA_MODEL": "llama3:8b",
        },
    ):
        c = Config(
            _env_file=None,  # type: ignore[call-arg]
        )
        assert c.ollama_base_url == "http://custom-host:9999"
        assert c.ollama_model == "llama3:8b"


# @lat: [[tests/config-tests#Config Tests#Case-Insensitive Variables]]
def test_config_case_insensitive() -> None:
    """Test that Config env vars are case-insensitive."""
    with patch.dict(
        "os.environ",
        {"ollama_model": "gemma:2b"},
    ):
        c = Config(
            _env_file=None,  # type: ignore[call-arg]
        )
        assert c.ollama_model == "gemma:2b"


# @lat: [[tests/config-tests#Config Tests#Extra Variables Ignored]]
def test_config_extra_ignore() -> None:
    """Test that Config ignores extra environment variables."""
    with patch.dict(
        "os.environ",
        {"UNKNOWN_SETTING": "should-not-fail"},
    ):
        c = Config(
            _env_file=None,  # type: ignore[call-arg]
        )
        assert not hasattr(c, "unknown_setting")


# @lat: [[tests/config-tests#Config Tests#Field Descriptions]]
def test_config_field_descriptions() -> None:
    """Test that Config fields have descriptions."""
    fields = Config.model_fields
    assert fields["ollama_base_url"].description is not None
    assert fields["ollama_model"].description is not None


# @lat: [[tests/config-tests#Config Tests#Module-Level Instance]]
def test_config_module_level_instance() -> None:
    """Test that the module-level config instance is a Config."""
    assert isinstance(config, Config)
    assert config.ollama_base_url  # Not empty
    assert config.ollama_model  # Not empty
