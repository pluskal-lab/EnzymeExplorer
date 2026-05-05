# AGENTS.md — EnzymeExplorer

This file provides **project-specific instructions for AI coding agents** (e.g., Cursor agents).
Follow these rules when reading, modifying, or adding code in this repository.

## What this repo is
**EnzymeExplorer** is a Python codebase for terpene synthase (TPS) discovery and downstream classification.
It supports:
- running predictions (full model requiring structures, and a sequence-only variant),
- data preparation and workflow steps used in the paper,
- model training/evaluation utilities and structural-domain analysis.

## Repository layout (high-level)
- `enzymeexplorer/` — main Python package (core logic lives here).
- `scripts/` — command-line / convenience scripts (e.g., `predict_*.py`, `bundle_fold_checkpoints.py`).
- `data/` — small, versioned artifacts for reproducibility (some `.pkl`, curated CSVs, etc.).
- `outputs/` — logs, intermediate artifacts, trained models/results (often large).
- `notebooks/` — notebooks / colab-related material.
- `build/` — generated build artifacts (do not edit manually).
- `app_faster_with_foldseek.py` — app/entry script related to Foldseek-accelerated workflow.

## Coding standards (must match CI)
Before proposing changes, run the same checks CI runs:

### Formatting
- `black --check .`

### Lint
- `flake8 . --max-line-length 88 --extend-ignore E203 --select B950`

### Pylint
- `pylint ./enzymeexplorer --disable C0301,E0401,E1136,R0914,R1728,R0912,R0917,R0915,R0913,R0902,W0640`

### Typing
- `mypy --implicit-optional --explicit-package-bases .`

### Additional agent check
- `ruff check .`

**Do not** reformat large unrelated parts of the repo. Keep diffs minimal and targeted.

## Python coding rules for agents
1. **Prefer minimal changes**
   - Make the smallest correct change that fixes the bug or adds the feature.
   - Do not mix unrelated refactors into the same patch.
   - Do not rename/move modules unless necessary for the requested change.

2. **Keep structure readable**
   - Prefer small helpers over deeply nested logic.
   - Reuse existing patterns in the repo before inventing new abstractions.
   - Keep control flow simple and explicit.
   - Choose clear names over clever code.

3. **Do not break public APIs**
   - Preserve function signatures, CLI arguments, file formats, and output contracts unless the task explicitly requires changing them.
   - If a public API change is unavoidable, stop and report it clearly before proceeding.
   - Maintain backward-compatible defaults whenever possible.

4. **Stop if ambiguous**
   - Do not guess silently about domain semantics, column names, file formats, status flags, or expected outputs.
   - If the task is ambiguous in a way that affects correctness, stop and report the exact questions.
   - Mirror existing project behavior whenever the repo already answers the question.

5. **Keep touched code ruff-clean**
   - Any file you modify should remain `ruff check` clean.
   - Also keep it compliant with Black, Flake8, Pylint, and Mypy as configured above.

6. **Avoid unnecessary memory usage**
   - Avoid loading more data than needed.
   - Stream, chunk, or slice when practical.
   - Prefer vectorized/filter-first approaches when they reduce memory and compute without obscuring correctness.

## Data / outputs hygiene
- Treat `outputs/` as **generated**. Do not commit large run products, logs, model artifacts, or downloaded bundles.
- Treat `data/` as **semi-static** (small reproducibility artifacts only). Avoid adding large datasets/binaries.
- If a workflow step needs big artifacts, prefer documenting download steps or paths rather than committing them.


## Performance / parallelism / caching
- Avoid heavy computation by default.
- Don’t kick off expensive training or massive database screening unless explicitly requested.
- Preserve reproducibility when adding caching or parallelism.
- Keep default behavior predictable and safe on modest hardware.

## Testing rules for agents
1. **Always add or adjust tests**
   - Any non-trivial behavior change should come with tests.
   - If fixing a bug, add a regression test whenever feasible.
   - If changing parsing/filtering/aggregation logic, test the changed behavior directly.

2. **Test pure logic explicitly**
   - Add focused unit tests for pure logic and helper functions, not just end-to-end scripts.
   - Prefer small synthetic inputs that isolate the behavior under test.
   - Test both expected-path and edge/failure-path behavior.

3. **Prefer synthetic/minimal fixtures**
   - Use synthetic data rather than large real artifacts whenever possible.
   - Keep tests deterministic, fast, and local.
   - Avoid network access and heavyweight external dependencies in tests.

4. **Preserve behavior unless intentionally changed**
   - When refactoring, ensure tests demonstrate no behavior regression.
   - If behavior changes intentionally, update tests to reflect the new contract.


## Working style for agents
1. **Start by locating the existing implementation**
   - Search in `enzymeexplorer/` first, then `scripts/`.

2. **Keep changes narrow**
   - Fix the bug / add the feature with the smallest correct diff.

3. **Don’t guess silently**
   - If a change depends on domain assumptions (e.g., file formats, column names, structure naming), check how the repo currently does it and mirror that.
   - If ambiguity remains and affects correctness, stop and report questions.

4. **Prefer adding small helpers over duplicating logic**
   - If you see repeated parsing/IO patterns, factor them into a utility module inside `enzymeexplorer/`.

5. **Avoid heavy computation by default**
   - Don’t kick off expensive training or massive database screening unless explicitly requested.

6. **Document user-facing changes**
   - If you modify defaults, CLI behavior, required inputs, or outputs, update docs/README accordingly.

If you modify any of the pipeline steps:
- keep reproducibility in mind,
- ensure existing default paths keep working,
- add/update docs for any new required inputs or outputs.

## Git / GitHub rules for agents
1. **Keep commits small and logical**
   - Prefer minimal, reviewable commits.
   - Do not mix formatting-only, refactor-only, and behavior-changing edits unless necessary.

2. **Preserve reviewability**
   - Keep diffs targeted and easy to inspect.
   - Summarize what changed, why, and how it was validated.

3. **Do not rewrite history unless explicitly asked**
   - Avoid force-push assumptions, rebases, or destructive history edits in instructions unless requested.

4. **Respect branch/review workflow**
   - Assume changes should be proposed through normal review flow.
   - Do not assume direct pushes to main/default branch are acceptable.

5. **Commit message style**
   - Use concise, informative commit messages.
   - Examples:
     - `fix: tighten sample validity filtering`
     - `test: add regression coverage for pure logic`
     - `refactor: extract reusable validity-mask helper`

## Required response format for agents
When proposing code changes, use this exact structure:

1. **Assumptions + questions**
   - Include only assumptions and questions that actually block correctness.

2. **Step-by-step plan**
   - Brief, concrete execution plan.

3. **Impacted files**
   - List paths that will change.

4. **Test plan**
   - Include exact commands to run.
   - State the MVP acceptance criteria in testable terms.
   - Prefer acceptance criteria that can be verified on synthetic data.

5. **Commit**
   - Provide a proposed commit message.

## PR checklist (for agents)
- [ ] Diff is minimal and targeted.
- [ ] Touched Python files are ruff-clean.
- [ ] Code formatted (Black).
- [ ] Flake8 passes.
- [ ] Pylint passes (with repo’s disables).
- [ ] Mypy passes (repo settings).
- [ ] Tests added/updated for changed behavior.
- [ ] Pure logic tests added/updated where applicable.
- [ ] No large artifacts committed (especially in `outputs/`).
- [ ] README/docs updated if user-facing behavior changed.
