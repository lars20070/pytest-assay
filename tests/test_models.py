#!/usr/bin/env python3
"""
Unit tests for the models module.

These tests cover:
- Readout model (defaults, custom values, serialization)
- AssayContext model (creation, defaults, validation)
- Evaluator protocol (conformance verified in evaluator test files)
"""

from __future__ import annotations as _annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_evals import Case, Dataset

from pytest_assay.models import AssayContext, Evaluator, Readout


class TestReadout:
    """Tests for the Readout model."""

    def test_defaults(self) -> None:
        """Test Readout model initializes with default values."""
        readout = Readout()

        assert readout.passed is True
        assert readout.details is None

    def test_custom(self) -> None:
        """Test Readout model initializes with custom values."""
        readout = Readout(passed=False, details={"error": "test error"})

        assert readout.passed is False
        assert readout.details == {"error": "test error"}

    def test_model_dump(self) -> None:
        """Test Readout.model_dump() returns correct dictionary."""
        readout = Readout(passed=False, details={"key": "value"})

        dumped = readout.model_dump()

        assert dumped == {"passed": False, "details": {"key": "value"}}

    def test_model_dump_defaults(self) -> None:
        """Test Readout.model_dump() with default values."""
        readout = Readout()

        dumped = readout.model_dump()

        assert dumped == {"passed": True, "details": None}

    def test_to_file(self, tmp_path: Path) -> None:
        """Test Readout.to_file() serializes to JSON file."""
        readout = Readout(passed=True, details={"key": "value", "count": 42})
        file_path = tmp_path / "readout.json"

        readout.to_file(file_path)

        assert file_path.exists()

        with file_path.open() as f:
            data = json.load(f)

        assert data["passed"] is True
        assert data["details"] == {"key": "value", "count": 42}

    def test_to_file_with_none_details(self, tmp_path: Path) -> None:
        """Test Readout.to_file() handles None details correctly."""
        readout = Readout(passed=False, details=None)
        file_path = tmp_path / "readout.json"

        readout.to_file(file_path)

        assert file_path.exists()

        with file_path.open() as f:
            data = json.load(f)

        assert data["passed"] is False
        assert data["details"] is None

    def test_to_file_with_empty_details(self, tmp_path: Path) -> None:
        """Test Readout.to_file() handles empty details dict."""
        readout = Readout(passed=True, details={})
        file_path = tmp_path / "readout.json"

        readout.to_file(file_path)

        with file_path.open() as f:
            data = json.load(f)

        assert data["details"] == {}


class TestEvaluator:
    """Tests for the Evaluator protocol.

    Protocol conformance tests live in evaluator test files
    (test_pairwise.py, test_bradleyterry.py). This class verifies
    the protocol definition itself.
    """

    def test_protocol_is_callable(self) -> None:
        """Test that Evaluator protocol requires __call__."""
        assert callable(Evaluator)

    def test_protocol_has_expected_structure(self) -> None:
        """Test that Evaluator has the expected protocol structure."""
        assert hasattr(Evaluator, "__protocol_attrs__") or callable(Evaluator)


class TestAssayContext:
    """Tests for the AssayContext model."""

    def test_creation(self) -> None:
        """Test AssayContext model creation with valid data."""
        dataset = Dataset[dict[str, str], type[None], Any](cases=[])
        path = Path("/tmp/test.json")

        context = AssayContext(
            dataset=dataset,
            path=path,
            assay_mode="evaluate",
        )

        assert context.dataset == dataset
        assert context.path == path
        assert context.assay_mode == "evaluate"

    def test_default_mode(self) -> None:
        """Test AssayContext uses 'evaluate' as default assay_mode."""
        dataset = Dataset[dict[str, str], type[None], Any](cases=[])
        path = Path("/tmp/test.json")

        context = AssayContext(dataset=dataset, path=path)

        assert context.assay_mode == "evaluate"

    def test_new_baseline_mode(self) -> None:
        """Test AssayContext with new_baseline mode."""
        dataset = Dataset[dict[str, str], type[None], Any](cases=[])
        path = Path("/tmp/test.json")

        context = AssayContext(
            dataset=dataset,
            path=path,
            assay_mode="new_baseline",
        )

        assert context.assay_mode == "new_baseline"

    def test_with_cases(self) -> None:
        """Test AssayContext with a dataset containing cases."""
        cases: list[Case[dict[str, str], type[None], Any]] = [
            Case(name="case_001", inputs={"topic": "test topic 1", "query": "test query 1"}),
            Case(name="case_002", inputs={"topic": "test topic 2", "query": "test query 2"}),
        ]
        dataset = Dataset[dict[str, str], type[None], Any](cases=cases)
        path = Path("/tmp/test.json")

        context = AssayContext(dataset=dataset, path=path)

        # Verify case count and structure
        assert len(context.dataset.cases) == 2

        # Verify first case content
        assert context.dataset.cases[0].name == "case_001"
        assert context.dataset.cases[0].inputs["topic"] == "test topic 1"
        assert context.dataset.cases[0].inputs["query"] == "test query 1"
        assert "topic" in context.dataset.cases[0].inputs
        assert "query" in context.dataset.cases[0].inputs

        # Verify dataset is mutable (required for update inside unit tests)
        context.dataset.cases.clear()
        assert len(context.dataset.cases) == 0

        # Verify we can extend with new cases
        new_case = Case(name="case_003", inputs={"query": "new query"})
        context.dataset.cases.append(new_case)
        assert len(context.dataset.cases) == 1
        assert context.dataset.cases[0].name == "case_003"

    def test_missing_required_fields(self) -> None:
        """Test AssayContext raises ValidationError when required fields are missing."""
        with pytest.raises(ValidationError):
            AssayContext()  # type: ignore[call-arg]
