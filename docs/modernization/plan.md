# Modernization Plan (2025-09-15)

This plan makes the project simpler, more modular, and aligned with current best practices while preserving your uv + Python 3.13 workflow and static CDN React.

## Targets
- Keep uv, pytest, ruff, mypy, loguru, pydantic.
- Improve backend modularity, validation, and error handling.
- Decide: stay on Flask or adopt FastAPI (typed, OpenAPI, pydantic-native).
- Clarify frontend path: stay with CDN React now; prepare migration path to Vite later.
- Add pre-commit hooks and CI.

## Backend: Recommended shape

Option A: Flask (minimal changes, lowest risk)
- Introduce app factory and Blueprints.
- Split routes by domain: `api/text`, `api/pdf`, `api/iris`, `api/image`, `pages`.
- Centralized error handlers + pydantic request models.
- Dependency layer: model loaders, clients, and services.

Option B: FastAPI (bigger win, moderate migration)
- Typed endpoints + OpenAPI + automatic validation.
- Uvicorn for dev, keep uv for env.
- Reuse services and model code; map Flask handlers → FastAPI routers.

Initial recommendation: Option A now, Option B later if needed.

### Proposed structure
```
src/
  server/
    app.py              # app factory create_app()
    __init__.py
    config.py           # pydantic BaseSettings
    extensions.py       # CORS, logging, etc.
    api/
      __init__.py
      text.py
      pdf.py
      iris.py
      image.py
    pages.py            # static page routes (home, image, csvTest)
  services/
    iris_service.py     # evaluate_iris_batch wrapper, caching
    pdf_service.py
    text_service.py
    image_service.py
```

### Key patterns
- Pydantic models for requests/responses.
- Centralized error handling via `app.errorhandler` or decorator.
- Caching via `functools.lru_cache` or `cachetools` keyed by parameters, not global state.
- Pathlib for all paths (already good).
- Config via `.env` + `pydantic-settings` (`BaseSettings`).

## Frontend: now vs later
- Now: keep CDN React, tidy directories: `static/home`, `static/image`, `static/csvTest` with ES modules per repo instruction.
- Later: Vite React app in `web/` with TypeScript + OpenAPI client generation from backend schema.

## Quality: tests, lint, types, CI
- Tests: add API tests using `pytest` and Flask test client.
- Lint/format: ruff (lint + format), mypy strict-ish on `src/`.
- Pre-commit: ruff, mypy, pytest -q (short), black if desired; keep ruff formatting if you prefer one tool.
- CI: GitHub Actions with `uv sync`, then run ruff, mypy, pytest.

## Phased plan

Phase 1 (today)
- Introduce create_app and Blueprints skeleton; move routes from `app.py` into `api/*.py` and `pages.py`.
- Add `server/config.py` with BaseSettings for CORS origins and limits.
- Add basic tests for one endpoint to lock behavior.

Phase 2
- Add pydantic request/response models for iris/text/pdf endpoints.
- Add error handlers and consistent JSON error schema.
- Extract services layer (`src/services`).

Phase 3
- Add pre-commit and GitHub Actions.
- Add OpenAPI docs (Flask via apispec/flask-smorest) OR consider migrating to FastAPI.

Phase 4 (optional)
- Frontend migration to Vite/React TS with generated API client from OpenAPI.

## Acceptance checklist
- `uv run run_app.py` launches server using `create_app()`.
- Routes are defined in Blueprints with pydantic validation wrappers.
- Tests for iris and text endpoints pass.
- Ruff and mypy pass on src/.
- CI green on main.
