#!/usr/bin/env python3
import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field
from pydantic_evals import Dataset
from pytest import Item


class Readout(BaseModel):
    """Result from an evaluator execution."""

    passed: bool = True
    details: dict[str, Any] | None = None

    def to_file(self, path: Path) -> None:
        """
        Serialize the readout to a JSON file.

        Args:
            path: The file path to write to.
        """
        with path.open("w") as f:
            json.dump(self.model_dump(), f, indent=2)


class Evaluator(Protocol):
    """
    Protocol for evaluation strategy callables.

    The Protocol defines ONLY what the plugin needs to call.
    Evaluator implementations configure themselves via __init__.
    """

    def __call__(self, item: Item) -> Coroutine[Any, Any, Readout]: ...


class AssayContext(BaseModel):
    """
    Context injected into assay tests containing dataset, path, and mode.

    Attributes:
        dataset: The evaluation dataset with test cases.
        path: File path for dataset persistence.
        assay_mode: "evaluate" or "new_baseline".
    """

    dataset: Dataset = Field(..., description="The evaluation dataset for this assay")
    path: Path = Field(..., description="File path where the assay dataset is stored")
    assay_mode: str = Field(default="evaluate", description='Assay mode: "evaluate" or "new_baseline"')
