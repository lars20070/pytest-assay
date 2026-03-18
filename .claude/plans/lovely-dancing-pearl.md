# Fix lat.md Knowledge Graph Problems 2–5

## Context

Following an audit of the `lat.md/` knowledge graph, four structural problems were identified. This plan addresses them in order of increasing scope.

---

## Problem 2 — `plugin-integration-tests.md`: no per-test H2 sections

**Files changed:**
- `lat.md/tests/plugin-integration-tests.md`
- `tests/test_plugin_integration.py`

**What:** The three integration tests are listed as bullets under the H1 with no sub-sections. Only `test_integration_pairwiseevaluator` has a `# @lat:` ref (pointing to the whole file). The other two tests have no ref at all.

**Fix:**
1. Rewrite `plugin-integration-tests.md` — replace the bullet list with three `##` sections:
   - `## PairwiseEvaluator Integration` — describes the creativity-criterion pairwise test. Links to `[[tests/test_plugin_integration.py#test_integration_pairwiseevaluator]]`.
   - `## BradleyTerryEvaluator Integration` — describes the BT tournament test with `max_standard_deviation=2.1`. Links to `[[tests/test_plugin_integration.py#test_integration_bradleyterryevaluator]]`.
   - `## Custom Evaluator Integration` — describes the user-defined `LengthEvaluator` test, demonstrating the extension point. Links to `[[tests/test_plugin_integration.py#test_integration_lengthevaluator]]`.
2. In `test_plugin_integration.py`: replace the coarse `# @lat: [[tests/plugin-integration-tests]]` on line 121 with three per-test refs:
   ```python
   # @lat: [[tests/plugin-integration-tests#Plugin Integration Tests#PairwiseEvaluator Integration]]
   # @lat: [[tests/plugin-integration-tests#Plugin Integration Tests#BradleyTerryEvaluator Integration]]
   # @lat: [[tests/plugin-integration-tests#Plugin Integration Tests#Custom Evaluator Integration]]
   ```

---

## Problem 3 — `tests/tests.md`: fragile short-form wiki links

**Files changed:**
- `lat.md/tests/tests.md`

**What:** All seven links use bare filenames (`[[plugin-unit-tests]]` etc.) which would become ambiguous if any other file with the same name were added.

**Fix:** Replace each link with its path-qualified form:

| Before | After |
|---|---|
| `[[plugin-unit-tests]]` | `[[tests/plugin-unit-tests]]` |
| `[[model-unit-tests]]` | `[[tests/model-unit-tests]]` |
| `[[bradleyterry-evaluator-tests]]` | `[[tests/bradleyterry-evaluator-tests]]` |
| `[[pairwise-evaluator-tests]]` | `[[tests/pairwise-evaluator-tests]]` |
| `[[config-tests]]` | `[[tests/config-tests]]` |
| `[[logger-tests]]` | `[[tests/logger-tests]]` |
| `[[plugin-integration-tests]]` | `[[tests/plugin-integration-tests]]` |

---

## Problem 4 — `evaluators.md` and `pytest-plugin.md`: `require-code-mention: true` on architecture docs

**Files changed:**
- `lat.md/evaluators.md`
- `lat.md/pytest-plugin.md`

**What:** Both files document architecture (protocols, lifecycle hooks, internal models), not test specs. `require-code-mention: true` is designed for test traceability. Applying it here treats every architecture section as a mandatory test target, conflating two distinct concerns.

**Fix:** Remove the frontmatter block (or set `require-code-mention: false`) from both files. The existing `# @lat:` refs in the source files can remain — they're still useful as documentation anchors; they just no longer need to be enforced by `lat check`.

Note: `lat check` currently passes because the source code happens to reference every leaf section. Removing the frontmatter simply stops `lat check` from enforcing this invariant on architecture docs going forward.

---

## Problem 5 — Missing cross-links between `pytest-plugin.md` and `evaluators.md`

**Files changed:**
- `lat.md/pytest-plugin.md`

**What:** `## _run_evaluation` constructs `EvaluatorInput`, invokes the `Evaluator` protocol, and serializes `Readout` — but never links to those concepts in `evaluators.md`. Readers following the plugin docs have no clickable path to the evaluator contracts.

**Fix:** Update the `## _run_evaluation` description in `pytest-plugin.md` to add inline wiki links:

```markdown
## _run_evaluation

Runs the evaluator against captured responses and serializes the resulting [[evaluators#Readout]] to `<assay_path>.readout.json`.

Retrieves the [[evaluators#Evaluator Protocol|evaluator]] callable from marker kwargs (default: `BradleyTerryEvaluator()`). Builds an [[evaluators#EvaluatorInput]] from the baseline snapshot and captured responses, then calls `asyncio.run(evaluator(eval_input))`.
```

---

## Verification

After all edits:
```bash
lat check   # must pass with 0 errors
```

No test changes are required for problems 3, 4, and 5 — those are pure documentation fixes. Problem 2 adds `# @lat:` refs to `test_plugin_integration.py` (no logic change).
