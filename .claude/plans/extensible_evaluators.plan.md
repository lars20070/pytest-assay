# Making `assays` evaluators extensible by end users

The best approach for the `assays` package combines a **Protocol-based interface** with **ABC-backed base class** and a **lightweight registry** — giving users three tiers of integration complexity while keeping the framework Pythonic and type-safe. This recommendation draws on analysis of six extensibility patterns and eight comparable frameworks, evaluated against the specific constraints of a pytest-integrated AI evaluation package.

## The current architecture and its constraint

Based on the package description, `assays` is a pytest plugin for evaluating Pydantic AI agents. It registers a custom marker `@pytest.mark.assay(evaluator=...)` and ships two built-in evaluators: **`PairwiseEvaluator`** (for pairwise comparison of outputs) and **`BradleyTerryEvaluator`** (implementing the Bradley-Terry probability model for ranking). The pytest plugin infrastructure likely processes the marker in a `conftest.py` or hook, instantiates the specified evaluator, and runs it against collected test results.

The core problem is that evaluator resolution is **closed to extension**. Users who want to create a custom evaluator — say, one that uses an LLM judge or a domain-specific scoring function — must fork the repository. This means the evaluator lookup or type-checking logic is hardcoded to recognize only the two built-in classes, with no public interface contract that external classes can satisfy. The `.claude/plans/custom_evaluators.plan.md` file captures this as a planned improvement.

## Six patterns evaluated for this use case

Each Python extensibility pattern was analyzed for fit with `assays`'s specific constraints: it's a **pytest plugin** (so it operates in the pytest collection/execution lifecycle), it handles **evaluator classes with state** (not just simple callables), and it must balance **type safety with minimal user friction**.

**Abstract Base Classes (ABC)** provide the strongest runtime guarantees. An `AbstractEvaluator` with `@abstractmethod` decorators on `evaluate()` and `score()` would cause `TypeError` at instantiation if a user forgets a required method — fail-fast, before any test runs. ABCs also enable shared logic via template methods (e.g., common result-formatting code in the base class). The downside is **mandatory inheritance**: users must `from assays import AbstractEvaluator` and subclass it, coupling their code to the framework.

**Python Protocols** (structural subtyping via `typing.Protocol`) offer the most flexible interface contract. A `@runtime_checkable class Evaluator(Protocol)` lets any class with the right methods satisfy the interface — no import or inheritance required. Static type checkers (mypy, Pyright) validate conformance at development time. The weakness: **`@runtime_checkable` only checks method existence, not signatures**, so a class with `evaluate(self, x)` instead of `evaluate(self, output, expected)` would pass `isinstance` checks but fail at call time.

**Entry points** (via `importlib.metadata`) are the Python packaging standard for cross-package plugin discovery — they're how pytest discovers its own 1,400+ plugins via the `pytest11` group. For `assays`, users would register evaluators in their `pyproject.toml` under an `assays.evaluators` group. This is **powerful for distributable plugins** but overkill for a user writing a one-off custom evaluator in their test directory.

**Factory pattern with registry** uses a decorator like `@register_evaluator("my_eval")` or `__init_subclass__` to build a name→class dictionary. This enables config-driven instantiation (`evaluator="my_eval"` as a string in the marker) but requires the module containing the registration to be imported first — a common source of bugs.

**Simple duck typing** — just documenting the expected interface and accepting any object — is Python's most traditional approach. Minimal framework code, maximum user freedom, zero type safety. Errors surface as `AttributeError` deep in test execution.

**Hook-based systems (pluggy)** power pytest itself and would integrate naturally with the existing plugin infrastructure. However, pluggy is designed for **lifecycle hooks** (pre/post events, result transformation) rather than **class-based extensibility**. Wrapping evaluator classes into a hook-based system would add significant conceptual overhead without corresponding benefit.

## What similar frameworks actually do

The most instructive comparisons come from frameworks solving nearly identical problems. **DeepEval**, an LLM evaluation framework, requires users to subclass `BaseMetric` and implement `measure()` and `a_measure()` methods — a pure ABC pattern with direct instantiation (no registry or discovery). **scikit-learn** takes the opposite approach: `make_scorer()` wraps any function with the right signature into a scorer object, requiring zero inheritance. **Hypothesis** uses functional composition via `@st.composite`, where custom strategies are just decorated functions that compose existing ones.

The pattern that maps closest to `assays` is **Hugging Face `evaluate`**, which uses ABC-style base class inheritance (`EvaluationModule` with `_info()`, `_compute()`, and `_download_and_prepare()`) plus Hub-based discovery for distribution. For a smaller-scale package, the **DeepEval model** is most relevant: a clear base class, direct instantiation, and automatic ecosystem integration (test infrastructure, reporting) simply by inheriting.

Notably, **no mature framework relies on duck typing alone** for extensibility. Every production framework uses either ABCs, Protocols, or at minimum a factory function to provide some interface contract. The trend since 2022 favors **Protocols for interface definitions** combined with **optional base classes for code reuse**.

## The recommended design: three tiers of integration

The optimal design for `assays` provides **three paths** for users, ordered by increasing sophistication. This mirrors how pytest itself works — from simple conftest fixtures to full entry-point plugins.

**Tier 1 — Subclass `BaseEvaluator` (recommended default).** Define an ABC that captures the evaluator contract and provides shared infrastructure:

```python
from abc import ABC, abstractmethod
from typing import Any

class BaseEvaluator(ABC):
    """Base class for all assay evaluators."""

    @abstractmethod
    def evaluate(self, results: list[dict[str, Any]]) -> EvaluationResult:
        """Evaluate collected test results and return a structured result."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable evaluator name."""
        ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register non-abstract subclasses
        if not getattr(cls, '__abstractmethods__', None):
            _evaluator_registry[cls.__name__] = cls
```

Users write:

```python
from assays import BaseEvaluator, EvaluationResult

class SemanticSimilarityEvaluator(BaseEvaluator):
    name = "semantic_similarity"

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def evaluate(self, results):
        # Custom scoring logic
        return EvaluationResult(score=0.92, passed=True)
```

This is the **primary recommended path** because it provides fail-fast validation (missing methods → `TypeError`), IDE autocomplete for required methods, and automatic registration via `__init_subclass__`. The `BaseEvaluator` can also contain shared formatting, logging, or result-aggregation logic that every evaluator benefits from.

**Tier 2 — Satisfy the `Evaluator` Protocol (for advanced users who don't want inheritance).** Define a companion Protocol that mirrors the ABC's interface:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Evaluator(Protocol):
    name: str
    def evaluate(self, results: list[dict[str, Any]]) -> EvaluationResult: ...
```

The framework's marker-processing code should type-check against this Protocol rather than the ABC:

```python
def _resolve_evaluator(evaluator) -> Evaluator:
    if isinstance(evaluator, type) and issubclass(evaluator, BaseEvaluator):
        return evaluator()  # Instantiate if class passed
    if isinstance(evaluator, Evaluator):  # Protocol check
        return evaluator
    raise TypeError(
        f"{type(evaluator).__name__} does not satisfy the Evaluator protocol. "
        f"It must have a 'name' attribute and an 'evaluate()' method."
    )
```

This lets users who prefer composition over inheritance — or who have an existing evaluator class from another library — use it directly without subclassing.

**Tier 3 — Entry-point discovery (for distributable plugin packages).** For users distributing evaluators as installable packages, support the `assays.evaluators` entry-point group:

```toml
# In the plugin's pyproject.toml
[project.entry-points."assays.evaluators"]
semantic_similarity = "my_evaluators:SemanticSimilarityEvaluator"
```

The framework discovers these at startup:

```python
from importlib.metadata import entry_points

def discover_plugin_evaluators():
    eps = entry_points(group="assays.evaluators")
    for ep in eps:
        cls = ep.load()
        _evaluator_registry[ep.name] = cls
```

This third tier is optional and should only be documented for advanced use cases. Most users will use Tier 1.

## Why this combination works best for `assays`

The **ABC + Protocol + entry-points** layered approach is not just theoretical best practice — it's what the most successful Python frameworks converge on. Four factors make it the right fit:

**Pytest integration is seamless.** The `__init_subclass__` auto-registration means that when a user defines a custom evaluator in their `conftest.py` (which pytest auto-imports), the evaluator is automatically available. No explicit registration step needed. The marker can accept either a class or an instance: `@pytest.mark.assay(evaluator=SemanticSimilarityEvaluator)` or `@pytest.mark.assay(evaluator=SemanticSimilarityEvaluator(threshold=0.9))`. The `_resolve_evaluator` function handles both cases.

**Type safety scales with user sophistication.** Beginners who subclass `BaseEvaluator` get IDE autocomplete showing exactly which methods to implement, plus instantiation-time errors if they forget one. Advanced users who use the Protocol get static type-checker validation without inheritance coupling. The Protocol's `@runtime_checkable` decorator provides a safety net at the framework boundary.

**It avoids the pluggy trap.** Since `assays` already runs as a pytest plugin, it might seem natural to use pluggy hooks for evaluator extensibility. But pluggy is designed for **event-driven hooks** (before/after test collection, before/after execution), not for **class-based domain objects**. Wrapping evaluators into hook implementations would force users to learn pluggy's `@hookspec`/`@hookimpl` system — a significant cognitive burden for what should be "write a class with an `evaluate()` method." Reserve pluggy for framework-level lifecycle hooks (e.g., `assays_before_evaluation`, `assays_after_evaluation`) where multiple plugins might participate, not for the evaluator interface itself.

**The migration path is minimal.** The existing `PairwiseEvaluator` and `BradleyTerryEvaluator` simply become subclasses of `BaseEvaluator`. No user-facing API changes are needed for existing code. The `@pytest.mark.assay(evaluator=PairwiseEvaluator)` syntax continues working exactly as before.

## Practical implementation checklist

The implementation should follow this order to keep the change manageable:

1. **Define `BaseEvaluator` ABC** with `evaluate()` and `name` as abstract members, plus `__init_subclass__` auto-registration. Extract the common interface from what `PairwiseEvaluator` and `BradleyTerryEvaluator` already share.
2. **Define the `Evaluator` Protocol** alongside the ABC, in the same module. Mark it `@runtime_checkable`.
3. **Refactor built-in evaluators** to subclass `BaseEvaluator`. Ensure backward compatibility — if the marker resolution currently checks `isinstance(evaluator, PairwiseEvaluator)`, change it to check the Protocol instead.
4. **Update the marker processing** in the pytest plugin to use `_resolve_evaluator()` with Protocol-based validation, accepting classes, instances, or string names (looked up in the registry).
5. **Add entry-point discovery** as an optional call in the plugin's `pytest_configure` hook: `discover_plugin_evaluators()`.
6. **Export the public API** explicitly: `from assays import BaseEvaluator, Evaluator, EvaluationResult` in `__init__.py`.
7. **Document all three tiers** with working examples, prioritizing Tier 1 (subclass) as the recommended approach.

## Conclusion

The extensibility problem in `assays` maps cleanly to a well-established design space in Python. The key insight is that **no single pattern is sufficient** — but the right combination of two or three patterns creates a system that is simultaneously easy for beginners (subclass and implement), flexible for experts (Protocol conformance), and scalable for the ecosystem (entry-point discovery). The ABC provides runtime safety and code reuse; the Protocol provides decoupling and static type safety; entry points provide cross-package distribution. This is the same layered approach used by scikit-learn (base classes + factory functions), MLflow (factory functions + entry-point plugins), and increasingly by modern Python libraries that have moved beyond pure duck typing toward **formalized-but-flexible** interfaces. For `assays` specifically, the `__init_subclass__` auto-registration is the key detail that makes the pytest integration frictionless — users define their evaluator in `conftest.py`, and it just works.