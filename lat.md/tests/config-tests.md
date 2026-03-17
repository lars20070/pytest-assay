---
lat:
  require-code-mention: true
---
# Config Tests

Tests in `tests/test_config.py` for the `Config` pydantic-settings model.

- Default values: `ollama_base_url="http://localhost:11434"`, `ollama_model="qwen2.5:14b"` when no env vars are set.
- Env override: `OLLAMA_BASE_URL` and `OLLAMA_MODEL` are read from environment variables.
- Case-insensitive: lowercase env var names are accepted.
- Extra env vars are silently ignored.
- Both fields have non-`None` descriptions.
- Module-level `config` instance is a `Config` with non-empty field values.
