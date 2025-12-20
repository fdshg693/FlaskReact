# Review Guideline (Current-State Codebase Review)

This document is a practical checklist for reviewing the **current state** of this repository (Flask + CDN React + Streamlit, ML/LLM utilities). It is not a PR review template; it is a playbook for assessing whether the codebase (or a specific file/module) follows the project’s documented rules.

## Severity labels

- **Blocker**: Must be fixed before merge (breaks run/build/tests, violates hard rules, unsafe behavior, or violates data/Git rules).
- **Major**: Should be fixed before merge (significant maintainability, correctness, or architectural issues).
- **Minor**: Nice to fix (small quality improvements, low-risk refactors, clearer naming).
- **Nit**: Style/wording preference, trivial polish.

If you are suggesting something not mandated by the repo docs, label it explicitly as **Recommended (non-blocking)**.

## Review flow (current-state)

1. **Define scope**
   - Review target: whole repo / a subsystem (Flask, ML, LLM, Streamlit) / a specific module.
   - Identify “must-follow” rules vs “recommended” practices.

2. **Run the minimum checks (as applicable)**
   - Install deps: `uv sync`
   - Run Flask app (required pattern): `uv run run_app.py`
     - **Blocker** if project documentation or scripts instruct running `src/server/app.py` directly.
   - Run tests: `uv run pytest`

3. **Scan and record findings**
   - Record findings using the severity labels (Blocker/Major/Minor/Nit).
   - For each finding, capture:
     - What rule is violated (link to the section in this document)
     - Location (module/path)
     - Suggested fix direction (smallest safe change)

4. **Summarize the health of the target**
   - Key risks (Blockers and Majors)
   - “Next refactors” list (Minors / Recommended)

## What to review (areas)

- Python standards (types, docstrings, pathlib, formatting)
- Data management (`data/original_sources/`, `data/outputs/`, `data/tmp/` + Git rules)
- Backend (Flask) API correctness
- Frontend (CDN React) integration
- Streamlit (if relevant)
- Security & secrets

## Code style: docstrings and comments

- **Blocker**: Every function and class must have a docstring.
- **Blocker**: Docstrings must follow **Google Style** and include (as applicable): **Args**, **Returns**, **Raises**.
- **Major**: If a function can raise, ensure `Raises:` is accurate and specific.

Comments:
- Add comments where intent is unclear.
- No comments are needed for self-explanatory code (but slightly “comment-heavy” is acceptable when it improves learning/readability).
- AI-generated English comments are acceptable; translating to Japanese is optional.

## Naming and structure

- Naming conventions:
	- `snake_case` for variables and functions
	- `PascalCase` for classes
	- `UPPER_SNAKE_CASE` for constants
	- Leading `_` for private/internal helpers
- **Major**: Use meaningful names; avoid `x`, `tmp`, and unclear abbreviations (only common ones like `url`, `api`, `db`).

Design:
- **Major**: Follow single responsibility (one function = one responsibility).
- **Major**: Split overly long functions.
- **Major**: Deduplicate repeated logic by extracting shared helpers where it improves maintainability.

## Python standards

- **Blocker**: Add type annotations to all functions (parameters and return types).
- Formatting:
	- **Major**: Prefer f-strings over `.format()` or `%` formatting.
- Filesystem:
	- **Blocker**: Use `pathlib.Path` for path/file operations; avoid `os.path`.

Modern libraries / patterns (from the coding standard table):
- **Major**: Prefer `dataclass` / `pydantic` over untyped “plain classes” for structured data.
- **Recommended (non-blocking)**: Prefer `typer` over `argparse` for CLI.
- **Recommended (non-blocking)**: Prefer `rich` over bare `print` for rich console output.
- **Recommended (non-blocking)**: Prefer `pydantic-settings` over directly using `os.environ` throughout the codebase.

## Refactoring / TODO markers

Refactoring policy (as documented):
- **Major**: Large improvements should not be done all at once; leave TODO comments and refactor incrementally.
- **Recommended (non-blocking)**: Keep TODO markers informative (priority/assignee/links) when helpful.

Supported markers:
- `TODO`: Future implementation
- `FIXME`: Works but needs fixing
- `HACK`: Temporary workaround (should be removed/refactored)
- `XXX`: Serious issue; needs urgent attention
- `NOTE`: Important explanation
- `OPTIMIZE`: Performance improvement opportunity

## Data management

Folder intent:
- `data/original_sources/`: primary/source data
- `data/outputs/`: generated/secondary data from processing
- `data/tmp/`: temporary, throwaway data

Placement rules:
- **Blocker**: Place data in the correct folder by type (source vs generated vs tmp).
- **Major**: Keep folder structure meaningful so it’s obvious which service/feature uses the data and what the output is for.
- **Major**: Track provenance of source data (e.g., a `README.md` describing where it came from).

Git inclusion rules:
- **Blocker**: Do not commit anything under `data/tmp/`.
- **Major**: Do not commit reproducible outputs under `data/outputs/` by default; include only when sharing is necessary and regeneration is costly.
- **Major**: Trained models should not be committed.

Large file guidance:
- **Major**: Any single file around **10MB+** should be treated as “needs review” to prevent repo bloat.
- **Major**: Large batches of images should not be committed; consider Git LFS.
- **Major**: If large data must be shared, use cloud storage / Git LFS / DVC and document download steps in a README.

## Testing strategy

This project does **not** aim for exhaustive coverage.

- **Blocker**: Changes must not break existing tests.
- **Major**: Add tests for:
	- Core/complex business logic
	- Complex calculations
	- Data transformation (parsing/normalization/format conversion)
	- Edge cases
	- Fragile refactor targets

Lower priority / usually not needed:
- Simple CRUD
- Trivial property existence checks (e.g., dataclass field existence)
- UI display logic
- “Thin wrapper” functions that only call an external API when mocking becomes disproportionately complex

Test placement:
- Tests live under `tests/` and should follow the existing directory grouping (e.g., `tests/server/`, `tests/machineLearning/`, `tests/config/`).
- Shared fixtures belong in `tests/conftest.py`.

Manual verification is valid:
- Use `if __name__ == "__main__":` blocks for quick module execution.
- Use `examples/` scripts or Jupyter notebooks when interactive validation is useful.

## Backend (Flask) review points

Startup pattern:
- **Blocker**: The Flask app must be started via `uv run run_app.py` (project-wide requirement; do not recommend running `src/server/app.py` directly).

API correctness:
- **Major**: Inputs must be validated (especially file uploads: size/type). Use stable, explicit error handling.
- **Major**: Return appropriate HTTP status codes; avoid silent failures.
- **Major**: JSON response shapes should be consistent and understandable for the frontend.

## Frontend (CDN React) review points

This repo uses **CDN React with no bundler**.

- **Major**: Keep the existing page structure conventions (static page folders with `index.html` + JS).
- **Major**: Use the established React import pattern: `const { useState, useEffect } = React;`.
- **Major**: Confirm API requests (endpoints, payloads such as `FormData`/base64) match backend expectations.
- **Minor**: Ensure the UI handles errors without silent failures.

## Streamlit review points

Streamlit apps are explicitly “experimental UIs”.

- **Major**: Changes should not break the ability to run Streamlit scripts.
- **Major**: Apply the same Python standards (type hints, pathlib, docstrings).
- **Major**: If the app depends on data files, ensure data management rules are followed (don’t commit temp outputs; document provenance for sources).

## Security and secrets

- **Blocker**: No secrets should be committed (API keys, tokens, credentials).
- **Major**: Avoid trusting client-provided filenames/paths; prevent path traversal when saving uploads.
- **Recommended (non-blocking)**: Avoid logging sensitive values (secrets, user-provided raw data) even in debug logs.

## Repository hygiene

- **Major**: Folder/module intent should remain clear (backend vs services vs ML vs LLM vs util).
- **Minor**: Naming and file organization should match existing patterns.
- **Nit**: Fix typos and clarify docstrings/comments when easy.

**Recommended (non-blocking)**: Maintain short “how to run/verify” notes in the most appropriate documentation for the target area.
