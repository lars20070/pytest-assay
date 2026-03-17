---
lat:
  require-code-mention: true
---
# Model Unit Tests

Tests for data models in `tests/test_models.py`.

## EvaluatorInput

Tests for the `EvaluatorInput` Pydantic model, covering field defaults, validation, and edge cases.

- Creation with `baseline_dataset=None` and a single response.
- Creation with a populated `Dataset` and verification of case count and names.
- Empty `agent_responses` list accepted.
- Multiple responses stored in order.
- `ValidationError` raised when all required fields are omitted.
- `ValidationError` raised when `agent_responses` is omitted.
- `ValidationError` raised when `baseline_dataset` is omitted.
- Exactly two model fields (`baseline_dataset`, `agent_responses`) are defined.
- A response with `output=None` is stored without error.

## Readout

Tests for the `Readout` model, covering default values, custom values, and file serialization.

- Default values: `passed=True`, `details=None`.
- Custom values accepted for both fields.
- `model_dump()` returns the correct dict for custom and default values.
- `to_file()` writes valid JSON; verifies `passed` and `details` fields in the file.
- `to_file()` with `details=None` writes `null`.
- `to_file()` with `details={}` writes an empty object.

## Evaluator Protocol

Tests verifying the `Evaluator` protocol is importable and has the expected callable structure.

- `Evaluator` is callable (protocol is importable and callable as a type).
- Protocol has the expected structure (`__protocol_attrs__` or callable).

## AssayContext

Tests for the `AssayContext` data model, covering field values, defaults, dataset mutability, and validation.

- Created with all required fields; verifies attribute values.
- Default `assay_mode` is `"evaluate"`.
- `assay_mode="new_baseline"` accepted.
- Dataset with cases: verifies case count, field access, mutability (cases can be cleared and extended).
- `ValidationError` raised when required fields are omitted.
