## Testing and Test Coverage

The tests are split into unit tests and integration tests. The integration tests require a local Ollama instance with the `qwen2.5:14b` model and are skipped in CI, where no GPU is available.

Locally, the test coverage is above 90%. In the CI pipeline, it is necessarily lower. Mocking the Ollama responses is possible, but not worth the effort at this point. We therefore accept the lower coverage in CI.