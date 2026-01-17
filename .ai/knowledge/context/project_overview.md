# FlaskReact AI Development Instructions

## Project Architecture

Experimental Flask + React (CDN) + Streamlit project for ML & LLM utilities.

- **Backend**: Flask server (`src/server/app.py`) launched via `src/run_app.py`
- **Frontend**: Static CDN React in `static/` (no bundler)
- **Core Modules**: `src/llm/`, `src/machineLearning/`, `src/util/`, `src/scrape/`, `src/streamlit/`
- **Data Flow**: `data/` → training (`src/machineLearning/`) → (planned unified model storage) → inference endpoints

## Critical Startup Pattern

**ALWAYS use `uv run run_app.py`** - never run `src/server/app.py` directly. The launcher:
- Configures Python path for proper module imports 
- Sets up project root environment
- Enables absolute imports like `from llm.image import analyze_image`

**ALWAYS use `uv run *.py`** for other scripts to ensure dependencies are correctly managed.

## Development Environment Standards

**Python Environment (3.13 + uv)**:
- Use `uv sync` for dependency management 
- All Python files follow modern patterns from `.github/instructions/modern.instructions.md`:
  - `pathlib.Path` for file operations (never `os.path`)
  - Type hints on all functions
  - `loguru` for logging (not built-in `logging`)
  - `pydantic` for data validation

**React Environment**:
- CDN React (no build) per `.github/instructions/react.instructions.md`
- Import pattern: `const { useState, useEffect } = React;`
- Pages in `static/{page}/` (`index.html` + `js/`)

**Streamlit**:
- Experimental UIs: `src/streamlit/agent_app.py`, `machine_learning.py`, `simple_app.py`

## Key Workflows

**Running the Application**:
```bash
uv sync                     # Install dependencies
uv run run_app.py          # Start Flask (localhost:8000)
```

**Machine Learning Pipeline**:
Train / evaluate:
```bash
uv run python -m src.machineLearning.ml_class          # Train Iris etc.
uv run python -m src.machineLearning.eval_batch        # Batch eval (if main provided)
```
Artifacts (models/scalers) currently stored ad-hoc; consolidation pending.

**API Endpoints Pattern**:
Pattern:
- File uploads: validation helper (`validate_file_upload()`)
- LLM: `/api/analyze-image`, `/api/extract-pdf-text`, `/api/split-text`
- ML: `/api/iris-prediction`, `/api/iris-batch-prediction`

## Module Organization

**`src/llm/`**:
- `image.py` (Vision API integration)
- `pdf.py` (PDF → text)
- `text_splitter.py` (LangChain chunking)

**`src/machineLearning/`**:
- `ml_class.py` (training)
- `eval_batch.py` (batch prediction)
- `save_util.py` (curves / CSV logs)

**`static/`**:
- `home/` (Iris single + batch)
- `image/` (image / pdf / text ops)
- `csvTest/` (upload tests)

## Testing & Code Quality

**Testing**:
- Use `pytest` for Python tests in `test/` directory
- Example: `test/util/test_convert_json.py` shows testing patterns

**AI-Assisted Development**:
- See `.github/explanation.md` for consolidated prompt usage
- Review workflow: create `review/` dir alongside target file → run review prompt → apply fix prompt

## Data Flow Patterns

**File Upload → Processing**:
1. Frontend sends FormData or base64 to Flask API
2. Flask validates with `validate_file_upload()`
3. Processing in respective modules (`llm/`, `machineLearning/`)
4. Results returned as JSON

**ML Inference** (current):
1. User input → `util.convert_json_to_model_input()`
2. Load trained model / scaler (path strategy TBD)
3. Predict → JSON response

## Security Considerations

Baseline:
- CORS limited (localhost dev)
- Upload validation (size/type)
- `werkzeug.secure_filename` usage
- Secrets via `.env` (see `sample.env`)

## Common Patterns to Follow

- Use absolute imports from project root (enabled by `run_app.py`)
- Error handling with appropriate HTTP status codes
- Logging with `loguru.logger` instead of print statements
- Type annotations for all function parameters and returns
- File operations exclusively with `pathlib.Path`
