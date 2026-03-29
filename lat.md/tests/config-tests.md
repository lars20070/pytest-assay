---
lat:
  require-code-mention: true
---
# Config Tests

Tests in [[tests/test_config.py]] for the `Config` pydantic-settings model.

## Default Values

`ollama_base_url` defaults to `http://localhost:11434` and `ollama_model` to `qwen2.5:14b` when no env vars are set. See [[tests/test_config.py#test_config_default_values]].

## Environment Variable Override

`OLLAMA_BASE_URL` and `OLLAMA_MODEL` environment variables override the defaults when set. See [[tests/test_config.py#test_config_env_override]].

## Case-Insensitive Variables

Lowercase env var names (e.g. `ollama_model`) are accepted and override field values identically to uppercase names. See [[tests/test_config.py#test_config_case_insensitive]].

## Extra Variables Ignored

Unknown environment variables are silently ignored and do not appear as attributes on the `Config` instance. See [[tests/test_config.py#test_config_extra_ignore]].

## Field Descriptions

Both `ollama_base_url` and `ollama_model` fields carry a non-`None` description string in `model_fields`. See [[tests/test_config.py#test_config_field_descriptions]].

## Module-Level Instance

The module-level `config` object is a `Config` instance with non-empty values for both fields. See [[tests/test_config.py#test_config_module_level_instance]].
