# Plan: Add `lat check` CI Job

## Context

The project uses [lat.md](https://www.npmjs.com/package/lat.md) to maintain a knowledge graph in `lat.md/`. The `lat check` command validates all wiki links and `@lat:` code references. Currently this is only run manually — adding it to CI ensures broken links are caught on every push/PR.

## Approach

Create a **separate workflow file** `.github/workflows/lat-check.yaml` (option 1b) so that `lat check` runs even on md-only PRs — the main `build.yaml` has `paths-ignore: **/*.md` which would skip it otherwise.

### Workflow definition

```yaml
name: lat check

on:
  push:
    branches:
      - master
  pull_request:

permissions:
  contents: read

jobs:
  lat-check:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout the repository
        uses: actions/checkout@v4.2.0
      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: "22"
      - name: Install lat.md
        run: npm install -g lat.md
      - name: Run lat check
        run: lat check
```

### Key decisions

- **Separate workflow file** — no `paths-ignore` on `**/*.md`, so md-only PRs trigger the check.
- **No dependency on `ci` job** — runs independently and in parallel.
- **No Python/uv needed** — `lat check` is a standalone Node.js CLI.
- **Node 22** — current LTS.

### File to create

- `.github/workflows/lat-check.yaml`

## Verification

- Push branch, open PR, confirm `lat check` job appears and passes.
- Introduce a broken `[[link]]` in a lat.md file, confirm CI fails.
