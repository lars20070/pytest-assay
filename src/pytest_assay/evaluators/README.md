## Evaluators

The *pytest-assay* framework provides two built-in evaluators:
* `PairwiseEvaluator` compares the pre-recorded baseline responses with the current agent responses in a pairwise manner. The *readout* report contains the boolean comparison results. See [integration test](../../../tests/pytest_assay/test_plugin_integration/test_integration_pairwiseevaluator.readout.json) for example.
* `BradleyTerryEvaluator` scores both baseline and current responses using the [Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley–Terry_model). The *readout* report contains the scores for each response and the overall ranking. See [integration test](../../../tests/pytest_assay/test_plugin_integration/test_integration_bradleyterryevaluator.readout.json) for example.
