---
lat:
  require-code-mention: true
---
# Plugin Integration Tests

End-to-end tests against a live Ollama instance in `tests/test_plugin_integration.py`. All marked `@pytest.mark.ollama` and `@pytest.mark.asyncio`.

A shared `generate_evaluation_cases()` generator produces ten research-topic cases. An agent generates search queries using a creative prompt. Three evaluators are exercised:

- `test_integration_pairwiseevaluator`: uses `PairwiseEvaluator` with a creativity criterion; verifies the full plugin lifecycle produces a `.readout.json` on disk.
- `test_integration_bradleyterryevaluator`: uses `BradleyTerryEvaluator` with the same creativity criterion and `max_standard_deviation=2.1`.
- `test_integration_lengthevaluator`: uses a user-defined `LengthEvaluator` (not part of the package) that passes when a majority of novel responses are longer than their baseline counterparts; demonstrates the custom evaluator extension point.
