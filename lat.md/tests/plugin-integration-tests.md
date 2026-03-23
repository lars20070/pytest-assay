---
lat:
  require-code-mention: true
---
# Plugin Integration Tests

End-to-end tests against a live Ollama instance in [[tests/test_plugin_integration.py]]. All marked `@pytest.mark.ollama` and `@pytest.mark.asyncio`.

A shared `generate_evaluation_cases()` generator produces ten research-topic cases. An agent generates search queries using a creative prompt.

## PairwiseEvaluator Integration

Runs the full plugin lifecycle with `PairwiseEvaluator` using a creativity criterion; verifies a `.readout.json` is written to disk. See [[tests/test_plugin_integration.py#test_integration_pairwiseevaluator]].

## BradleyTerryEvaluator Integration

Runs the full plugin lifecycle with `BradleyTerryEvaluator` using the same creativity criterion and `max_standard_deviation=2.1`; verifies a `.readout.json` is written to disk. See [[tests/test_plugin_integration.py#test_integration_bradleyterryevaluator]].

## Custom Evaluator Integration

Runs the full plugin lifecycle with a user-defined `LengthEvaluator` (not part of the package) that passes when a majority of novel responses are longer than their baseline counterparts. Demonstrates the custom evaluator extension point. See [[tests/test_plugin_integration.py#test_integration_lengthevaluator]].
