#!/usr/bin/env python3
"""Shared Ollama constants for test configuration."""

import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "qwen2.5:14b"
