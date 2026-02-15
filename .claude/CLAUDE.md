# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pytest-assay is a framework for the evaluation of Pydantic AI agents. By adding the `@pytest.mark.assay` decorator to a test, you can run an assay resulting in a readout report. The assay compares the current agent responses against previously recorded baseline responses, e.g. from the main branch. The implementation is based on pytest hooks which capture `Agent.run()` responses. 

## Development Commands

### Environment Setup

```bash
# Install Ollama (required for local models)
# See https://ollama.com for installation instructions

# Pull the default model
ollama pull qwen3:8b

# Install dependencies
uv sync
```

### Testing

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_utils.py

# Run specific test
uv run pytest tests/test_utils.py::test_duckduckgo_search

# Run tests in parallel
uv run pytest -n auto

# Run tests with coverage report
uv run pytest --cov=src/pytest-assay --cov-report=term-missing
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Check and fix linting issues (ALWAYS run with --fix)
uv run ruff check --fix .

# Type checking (ALWAYS run after code changes)
uv run pyright .
```

### Before Committing

Run these checks:

```bash
# 1. Format code
uv run ruff format .

# 2. Check and fix linting issues
uv run ruff check --fix .

# 3. Type checking
uv run pyright .

# 4. Run tests
uv run pytest -n auto
```

## MCP Servers

This project uses Model Context Protocol (MCP) servers to extend AI capabilities. These are automatically invoked when relevant.

### Context7 Documentation Server

**When to use:**
- Looking up library documentation (e.g., "How do I use pydantic-ai streaming?")
- Checking API references for dependencies
- Finding code examples from official docs
- Verifying correct usage of third-party packages

**Examples:**
- "What's the latest pydantic-ai agent syntax?"
- "Show me httpx async client examples"
- "How do I configure pytest-asyncio?"

### GitHub Repository Server

**When to use:**
- Checking open/closed issues in this repository
- Reviewing pull requests and their status
- Reading issue comments and discussions
- Finding related issues or PRs
- Understanding project history and decisions

**Examples:**
- "What are the open issues about curiosity?"
- "Show me recent PRs related to PDF support"
- "Are there any issues about MLX integration?"
- "What's the status of issue #13?"

### Best Practices

- **Be specific:** "Check issue #15" is better than "check issues"
- **Context first:** Read codebase before checking issues
- **Combine sources:** Use Context7 for "how to use X" and GitHub for "what's our approach to X"

## Coding Standards

See the rules files for detailed coding standards:

- `.claude/rules/python.md` - Python coding standards, type hints, async patterns
- `.claude/rules/python-testing.md` - Testing conventions, markers, coverage requirements
