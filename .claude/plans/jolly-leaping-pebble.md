# Plan: Top 3 lat.md Knowledge Base Improvements

## Context

`lat check` passes and all 11 markdown files are structurally valid (leading paragraphs, wiki links, code refs). However, two source classes lack architecture documentation entirely, and the knowledge graph has no cross-links from architecture docs to test specs — making it hard to navigate between "what does X do?" and "how is X tested?".

---

## Improvement 1 — Document `Config` in architecture

**Problem:** `src/pytest_assay/config.py` contains the central `Config` pydantic-settings model (fields, env var overrides, defaults) but there is zero architecture documentation for it. `config-tests.md` covers test specs but captures no design intent.

**Fix:**
- Add `## Config` section to `lat.md/pytest-plugin.md` documenting: fields (`assay_mode`, `baseline_dir`, `model`, etc.), env var prefix, defaults, and why pydantic-settings was chosen.
- Add `# @lat: [[pytest-plugin#Config]]` to the `Config` class in `src/pytest_assay/config.py`.

**Files:** `lat.md/pytest-plugin.md`, `src/pytest_assay/config.py`

---

## Improvement 2 — Document `AssayContext` in architecture

**Problem:** `AssayContext` is a pydantic model in `models.py` that carries captured agent responses through the plugin lifecycle (created in setup, populated in call, consumed in teardown). It has test specs in `model-unit-tests.md` but no architecture section explaining its role, fields, or design rationale.

**Fix:**
- Add `## AssayContext` section to `lat.md/pytest-plugin.md` (alongside other plugin models like Stash Keys) explaining: what it holds, when it's created, how it flows through the lifecycle via stash, and why it's a pydantic model.
- Add `# @lat: [[pytest-plugin#AssayContext]]` to the `AssayContext` class in `src/pytest_assay/models.py`.

**Files:** `lat.md/pytest-plugin.md`, `src/pytest_assay/models.py`

---

## Improvement 3 — Add architecture → test spec cross-links

**Problem:** `evaluators.md` and `pytest-plugin.md` have no links to their corresponding test spec files. A reader of the architecture cannot navigate to "how is this tested?" — breaking the knowledge graph's navigability.

**Fix:** Add `See also` lines (wiki links) near the top or in relevant sections of each architecture file pointing to test specs:
- `evaluators.md`: link to `[[tests/bradleyterry-evaluator-tests]]`, `[[tests/pairwise-evaluator-tests]]`, `[[tests/model-unit-tests]]`
- `pytest-plugin.md`: link to `[[tests/plugin-unit-tests]]`, `[[tests/plugin-integration-tests]]`

Conversely, add reverse links from test spec index (`tests/tests.md`) back to architecture files.

**Files:** `lat.md/evaluators.md`, `lat.md/pytest-plugin.md`, `lat.md/tests/tests.md`

---

## Verification

After each improvement:
```bash
lat check          # must still pass (no broken links)
lat refs "pytest-plugin#Config"           # should show config.py
lat refs "pytest-plugin#AssayContext"     # should show models.py
```
