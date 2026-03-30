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

%% lat:begin %%
# Before starting work

- Run `lat search` to find sections relevant to your task. Read them to understand the design intent before writing code.
- Run `lat expand` on user prompts to expand any `[[refs]]` — this resolves section names to file locations and provides context.

# Post-task checklist (REQUIRED — do not skip)

After EVERY task, before responding to the user:

- [ ] Update `lat.md/` if you added or changed any functionality, architecture, tests, or behavior
- [ ] Run `lat check` — all wiki links and code refs must pass
- [ ] Do not skip these steps. Do not consider your task done until both are complete.

---

# What is lat.md?

This project uses [lat.md](https://www.npmjs.com/package/lat.md) to maintain a structured knowledge graph of its architecture, design decisions, and test specs in the `lat.md/` directory. It is a set of cross-linked markdown files that describe **what** this project does and **why** — the domain concepts, key design decisions, business logic, and test specifications. Use it to ground your work in the actual architecture rather than guessing.

# Commands

```bash
lat locate "Section Name"      # find a section by name (exact, fuzzy)
lat refs "file#Section"        # find what references a section
lat search "natural language"  # semantic search across all sections
lat expand "user prompt text"  # expand [[refs]] to resolved locations
lat check                      # validate all links and code refs
```

Run `lat --help` when in doubt about available commands or options.

If `lat search` fails because no API key is configured, explain to the user that semantic search requires a key provided via `LAT_LLM_KEY` (direct value), `LAT_LLM_KEY_FILE` (path to key file), or `LAT_LLM_KEY_HELPER` (command that prints the key). Supported key prefixes: `sk-...` (OpenAI) or `vck_...` (Vercel). If the user doesn't want to set it up, use `lat locate` for direct lookups instead.

# Syntax primer

- **Section ids**: `lat.md/path/to/file#Heading#SubHeading` — full form uses project-root-relative path (e.g. `lat.md/tests/search#RAG Replay Tests`). Short form uses bare file name when unique (e.g. `search#RAG Replay Tests`, `cli#search#Indexing`).
- **Wiki links**: `[[target]]` or `[[target|alias]]` — cross-references between sections. Can also reference source code: `[[src/foo.ts#myFunction]]`.
- **Source code links**: Wiki links in `lat.md/` files can reference functions, classes, constants, and methods in TypeScript/JavaScript/Python/Rust/Go/C files. Use the full path: `[[src/config.ts#getConfigDir]]`, `[[src/server.ts#App#listen]]` (class method), `[[lib/utils.py#parse_args]]`, `[[src/lib.rs#Greeter#greet]]` (Rust impl method), `[[src/app.go#Greeter#Greet]]` (Go method), `[[src/app.h#Greeter]]` (C struct). `lat check` validates these exist.
- **Code refs**: `// @lat: [[section-id]]` (JS/TS/Rust/Go/C) or `# @lat: [[section-id]]` (Python) — ties source code to concepts

# Test specs

Key tests can be described as sections in `lat.md/` files (e.g. `tests.md`). Add frontmatter to require that every leaf section is referenced by a `// @lat:` or `# @lat:` comment in test code:

```markdown
---
lat:
  require-code-mention: true
---
# Tests

Authentication and authorization test specifications.

## User login

Verify credential validation and error handling for the login endpoint.

### Rejects expired tokens
Tokens past their expiry timestamp are rejected with 401, even if otherwise valid.

### Handles missing password
Login request without a password field returns 400 with a descriptive error.
```

Every section MUST have a description — at least one sentence explaining what the test verifies and why. Empty sections with just a heading are not acceptable. (This is a specific case of the general leading paragraph rule below.)

Each test in code should reference its spec with exactly one comment placed next to the relevant test — not at the top of the file:

```python
# @lat: [[tests#User login#Rejects expired tokens]]
def test_rejects_expired_tokens():
    ...

# @lat: [[tests#User login#Handles missing password]]
def test_handles_missing_password():
    ...
```

Do not duplicate refs. One `@lat:` comment per spec section, placed at the test that covers it. `lat check` will flag any spec section not covered by a code reference, and any code reference pointing to a nonexistent section.

# Section structure

Every section in `lat.md/` **must** have a leading paragraph — at least one sentence immediately after the heading, before any child headings or other block content. The first paragraph must be ≤250 characters (excluding `[[wiki link]]` content). This paragraph serves as the section's overview and is used in search results, command output, and RAG context — keeping it concise guarantees the section's essence is always captured.

```markdown
# Good Section

Brief overview of what this section documents and why it matters.

More detail can go in subsequent paragraphs, code blocks, or lists.

## Child heading

Details about this child topic.
```

```markdown
# Bad Section

## Child heading

Details about this child topic.
```

The second example is invalid because `Bad Section` has no leading paragraph. `lat check` validates this rule and reports errors for missing or overly long leading paragraphs.
%% lat:end %%
