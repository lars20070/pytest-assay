# Design Analysis: `generator` in `@pytest.mark.assay` vs. External Dataset Creation

## Current Design

The `generator` callable is passed as a marker kwarg. The plugin calls it inside `pytest_runtest_setup` only if no serialized dataset file exists yet. The resulting `Dataset` is wrapped into an `AssayContext` and injected into the test via `item.funcargs["context"]`.

---

## Option A: Keep `generator` in the Marker (current)

**Pros:**
- **Declarative, self-contained test definition.** The marker is a single source of truth: evaluator *and* data source live together. You can read the decorator and understand the full assay without chasing fixtures.
- **Lazy, conditional generation.** The plugin only calls the generator when no baseline file exists. This logic is invisible to the test author -- they don't need `if not path.exists()` boilerplate.
- **Symmetry with `evaluator`.** Both the data source and the evaluation strategy are declared in the same place. Moving one out breaks that symmetry.

**Cons:**
- **Generator signature is constrained.** It must be a zero-argument callable returning `Dataset`. If a generator needs runtime information (e.g. fixtures, config, parametrize values), there's no clean way to pass it.
- **Magic injection of `context`.** The test receives `context: AssayContext` as a parameter, but it's not a real pytest fixture -- it's stuffed into `funcargs` by a hook. This is surprising to pytest users who expect fixtures to be defined via `@pytest.fixture` or `conftest.py`.
- **Harder to share datasets.** Two tests that share the same dataset must both reference the same generator function. With fixtures, you'd just depend on the same fixture name.

---

## Option B: Dataset Created in a Fixture / Test Body

**Pros:**
- **Standard pytest idiom.** A `@pytest.fixture` returning `AssayContext` is immediately understandable. No hidden `funcargs` injection.
- **Full access to other fixtures.** A fixture can depend on `tmp_path`, database connections, parametrize values, or any other fixture -- something a zero-arg generator cannot do.
- **Easier dataset sharing.** Multiple tests reuse the same fixture by name; no need to import and wire the same callable.

**Cons:**
- **Loses conditional loading.** The plugin currently skips generation when a baseline file already exists. Moving generation to a fixture means either (a) the fixture must replicate that logic, or (b) the plugin still needs a hook to intercept and replace the dataset -- defeating the purpose.
- **Splits the assay definition.** The marker would declare the evaluator, but the data source would live elsewhere (fixture or test body). You'd need to look in two places to understand a single assay.
- **Test body pollution.** If the dataset is built inside the test function, the test mixes setup concerns (data generation) with execution concerns (running the agent). Fixtures avoid this, but then you're back to the "two places" problem.

---

## Option C: Hybrid -- Fixture-Based with Marker Override

The marker could accept an *optional* `generator`. If absent, the plugin looks for a `context` fixture (standard pytest resolution). If present, the marker generator takes priority.

**Pros:** Best of both worlds -- simple cases stay declarative, complex cases use fixtures.
**Cons:** Two code paths to maintain and document. Users must understand when each applies.

---

## Conclusion

The current design is the right call for this project. The key reason: **the plugin needs control over *when* and *whether* to generate the dataset** (skip if baseline file exists, generate otherwise). That conditional logic is core to the assay lifecycle and belongs in the plugin, not in user-written fixtures. Forcing fixture authors to replicate that `if file.exists()` check would be error-prone and leak implementation details.

The zero-arg constraint on the generator is acceptable because dataset generation is inherently a pure function -- it defines *what to evaluate*, not *how to run* the test. If you ever need runtime-dependent datasets (e.g. parametrized topics), the cleaner extension would be to let the generator accept an optional config dict via the marker (`generator_kwargs={"n_cases": 5}`) rather than moving to fixtures.

The one thing worth improving: document that `context` is injected by the plugin, not a regular fixture. A brief note in the marker docstring or a type stub would reduce confusion.
