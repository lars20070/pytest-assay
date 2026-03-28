# Plan: `lat check` GitHub Action

## Context

The `lat.md` tool validates wiki links and code references via `lat check`, but there's no way for consumers to run this in CI. Projects like pytest-assay skip `.md` files in CI entirely. A reusable GitHub Action in the lat.md repo lets any project add `lat check` to their pipeline with one `uses:` line.

## Deliverable

A **composite GitHub Action** at `lat.md/.github/actions/check/action.yml`, usable as:

```yaml
uses: 1st1/lat.md/.github/actions/check@main
```

## Action design

### Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `scope` | `""` (all) | Optional: `md`, `code-refs`, `index`, `sections` |
| `node-version` | `22` | Node.js version for setup |
| `lat-version` | `latest` | lat.md npm version to install |

### Steps (composite)

1. `actions/setup-node@v4` with `node-version` input
2. `npm install -g lat.md@${{ inputs.lat-version }}`
3. Run `lat check ${{ inputs.scope }}`

Exit code propagation is automatic — `lat check` exits 1 on failure, which fails the step.

### Files to create/modify

**In `/Users/lars/Code/lat.md/`:**

1. **New:** `.github/actions/check/action.yml` — the composite action
2. **Modify:** `.github/workflows/ci.yml` — add a job that dogfoods the action on lat.md's own `lat.md/` folder

**In `/Users/lars/Code/pytest-assay/`:**

3. **Modify:** `.github/workflows/build.yaml` — add a `lat-check` job that uses the new action, triggered on `lat.md/**` path changes (inverse of the current `paths-ignore: **/*.md`)

## Key decisions

- **Composite over JavaScript action**: no build step, no `dist/` to maintain, ~15 lines of YAML
- **`npm install -g`** over `npx`: avoids re-downloading on every step if action is used multiple times; also makes the `lat` binary available for subsequent steps
- **Separate job in pytest-assay** (not added to the existing `ci` matrix): lat check is Node-based, independent of Python version matrix, and should trigger on `.md` changes (currently excluded)

## Publishing

Composite actions inside a repo are published automatically — no build or release artifact needed. Consumers reference them by repo path + git ref.

### Versioning with tags

After merging the action to `main`, create a semver tag:

```bash
git tag -a v0.1.0 -m "lat check GitHub Action"
git push origin v0.1.0
```

Then maintain a floating major-version tag (standard GitHub Action convention):

```bash
git tag -fa v0 -m "Update v0 to v0.1.0"
git push origin v0 --force
```

Consumers can then pin to a major version:

```yaml
uses: 1st1/lat.md/.github/actions/check@v0
```

On each new release (e.g. `v0.2.0`), move the `v0` tag forward.

### GitHub Marketplace (optional, later)

To list on the Marketplace, add `branding` to `action.yml`:

```yaml
branding:
  icon: 'check-circle'
  color: 'green'
```

Then go to the repo's Releases page → "Draft a new release" → check "Publish this action to the GitHub Marketplace". Requires the action to be in the repo root or a recognized path, and the repo must be public.

**Note:** Marketplace listing requires the `action.yml` to be at the repo root. Since ours is at `.github/actions/check/action.yml`, Marketplace listing would require either moving it to the repo root or creating a dedicated `1st1/lat-check-action` repo. This is a future consideration — the action works without Marketplace listing.

## Verification

1. In lat.md repo: `cd /Users/lars/Code/lat.md && lat check` should pass
2. In pytest-assay repo: `cd /Users/lars/Code/pytest-assay && lat check` should pass
3. Push a branch to lat.md, open a PR — CI should run the dogfood job
4. Push a branch to pytest-assay with a `lat.md/` change — the new `lat-check` job should trigger
