# Project Overview

Pytest-assay is a fully local web research and report writing assistant that prioritizes user privacy. It leverages PydanticAI as the core agent framework to orchestrate a multi-step research workflow. The system runs AI models locally using Ollama and performs web searches through SearXNG, ensuring no data leaves the user's environment by default. The application supports both local and cloud-based configurations, allowing users to choose between complete privacy or enhanced performance with API-based services.

The research workflow follows a state-machine pattern with four main stages:
1. **Web Search**: Initiates targeted searches based on the research topic
2. **Summarize Search Results**: Processes and condenses the retrieved information
3. **Reflect on Summary**: Evaluates the quality and completeness of gathered information
4. **Finalize Summary**: Produces the final research report with citations

This project uses PydanticAI's agent capabilities for structured responses, dependency injection, and tool integration. The UV package manager handles Python dependencies and virtual environment management.

## Folder Structure

```
pytest-assay/
├── benchmarks
│   ├── codenames
│   │   ├── task_schema.json
│   │   └── task.json
│   ├── dark_humor_detection
│   │   ├── task_schema.json
│   │   └── task.json
│   ├── README.md
│   └── rephrase
│       ├── task_schema.json
│       └── task.json
├── pytest-assay.log
├── dist
├── LICENSE
├── pyproject.toml
├── README.md
├── reports
├── src
│   └── pytest-assay
│       ├── __init__.py
│       ├── agents.py
│       ├── cli.py
│       ├── config.py
│       ├── evals
│       │   ├── evals.py
│       │   ├── import_bigbench.py
│       │   └── README.md
│       ├── examples.py
│       ├── graph.py
│       ├── logger.py
│       ├── models.py
│       ├── prompts.py
│       ├── py.typed
│       └── utils.py
├── tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── data
│   │   ├── state_1.json
│   │   ├── state_2.json
│   │   └── state_3.json
│   ├── README.md
│   ├── reports
│   │   ├── petrichor.md
│   │   └── petrichor.pdf
│   ├── test_config.py
│   ├── test_example.py
│   ├── test_graph.py
│   ├── test_logger.py
│   └── test_utils.py
├── uml
│   ├── classes.dot
│   ├── classes.png
│   ├── packages.dot
│   ├── packages.png
│   └── README.md
└── uv.lock
```

## Libraries and Frameworks

### Core Dependencies
- **pydantic-ai**: Main agent framework for building the research assistant
- **pydantic**: Data validation and settings management using Python type annotations
- **ollama**: Python client for interacting with local Ollama models

### Development Tools
- **python-dotenv**: Load environment variables from .env files
- **rich**: Terminal formatting for better CLI output
- **loguru**: Advanced logging with structured output
- **pytest**: Testing framework
- **pytest-asyncio**: Async test support for PydanticAI agents
- **ruff**: Fast Python linter and formatter

### Environment Management
- **uv**: Modern Python package and project manager
- **docker**: Container runtime for SearXNG deployment

## Coding Standards

### Python Style Guidelines
- Follow PEP 8 with a line length limit of 100 characters
- Use Python 3.11+ features including type hints for all function signatures
- Prefer async/await patterns for I/O operations and agent interactions
- Use Ruff for automatic code formatting and linting

### PydanticAI Agent Patterns
- Define agents with explicit `deps_type` and `output_type` for type safety
- Use dependency injection for passing configuration and services to agents
- Implement structured outputs using Pydantic models with Field descriptions
- Handle tool registration through PydanticAI's tool decorator pattern
- Use RunContext for accessing dependencies within agent functions

### Code Organization
- Keep agent definitions focused on a single responsibility
- Separate tool implementations from agent logic
- Use Pydantic models for all data structures and API contracts
- Implement proper error handling with custom exception classes
- Log all agent decisions and tool calls for debugging

### Testing Requirements
- Write unit tests for all agent tools and utility functions
- Use pytest fixtures for agent setup and teardown
- Mock external services (Ollama, SearXNG) in tests using pytest-mock
- Or use VCR recordings with pytest-recording
- Test both success and failure paths for agent workflows
- Maintain test coverage above 80% for core functionality

### Environment Configuration
- Never commit `.env` files or API keys to version control
- Use environment variables for all configuration values
- Provide sensible defaults for optional settings
- Document all required environment variables in `.env.example`
- Support both local (Ollama, SearXNG) and cloud (OpenAI, Anthropic) configurations

### Documentation Standards
- Use docstrings for all classes, methods, and functions
- Include type hints in function signatures
- Document agent system prompts and tool descriptions clearly
- Maintain up-to-date README with setup instructions
- Include examples of agent usage and output formats

### Error Handling
- Implement retry logic for network requests and LLM calls
- Use PydanticAI's built-in error handling for agent failures
- Provide meaningful error messages for user-facing issues
- Log errors with full context for debugging
- Gracefully degrade when optional services are unavailable

### Security Considerations
- Default to local-only operation without external API calls
- Validate and sanitize all user inputs
- Use HTTPS for any external API connections
- Store sensitive configuration in environment variables only
- Implement rate limiting for external service calls

### Performance Optimization
- Use async operations for concurrent web searches
- Implement caching for repeated searches within a session
- Stream LLM responses when possible for better UX
- Batch process search results for efficient summarization
- Monitor and log agent execution times for optimization

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
