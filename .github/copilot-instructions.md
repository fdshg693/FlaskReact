# FlaskReact AI Development Instructions

## Project Architecture

This is an experimental Flask+React web application for image analysis and machine learning operations. The architecture follows a clear separation:

- **Backend**: Flask server (`server/app.py`) with modular LLM and ML functionality
- **Frontend**: React via CDN (no build system) in `static/` with page-specific directories
- **Core Modules**: `llm/`, `machineLearning/`, `util/`, `scrape/`
- **Data Pipeline**: `data/` → model training → `param/`+`scaler/` → web inference

## Critical Startup Pattern

**ALWAYS use `uv run run_app.py`** - never run `server/app.py` directly. The `run_app.py` launcher:
- Configures Python path for proper module imports 
- Sets up project root environment
- Enables absolute imports like `from llm.image import analyze_image`

## Development Environment Standards

**Python Environment (3.13 + uv)**:
- Use `uv sync` for dependency management 
- All Python files follow modern patterns from `.github/instructions/modern.instructions.md`:
  - `pathlib.Path` for file operations (never `os.path`)
  - Type hints on all functions
  - `loguru` for logging (not built-in `logging`)
  - `pydantic` for data validation

**React Environment**:
- CDN-based React (no JSX compilation) following `.github/instructions/react.instructions.md`
- Import pattern: `const { useState, useEffect } = React;`
- Components in `static/{pageName}/js/` with HTML entry points

## Key Workflows

**Running the Application**:
```bash
uv sync                    # Install dependencies
uv run run_app.py         # Start server on localhost:8000
```

**Machine Learning Pipeline**:
- Train models: `python3 -m machineLearning.ml_class`
- Models saved to `param/`, scalers to `scaler/`  
- Batch evaluation via `machineLearning.eval_batch.evaluate_iris_batch()`

**API Endpoints Pattern**:
- File uploads use `validate_file_upload()` for security
- LLM operations: `/api/analyze-image`, `/api/extract-pdf-text`, `/api/split-text`
- ML inference: `/api/iris-prediction` (single), `/api/iris-batch-prediction`

## Module Organization

**`llm/` - LLM Operations**:
- `image.py`: Image analysis via OpenAI Vision
- `pdf.py`: PDF text extraction  
- `text_splitter.py`: Text chunking with LangChain
- All functions expect base64 data or file paths

**`machineLearning/` - ML Pipeline**:
- `ml_class.py`: Model training and persistence
- `eval_batch.py`: Batch prediction with preprocessing
- `save_util.py`: Learning curve visualization and CSV logging
- Models use scikit-learn with joblib persistence

**`static/` Frontend Structure**:
- `home/`: Iris prediction (single/batch)
- `image/`: Image analysis, PDF extraction, text splitting
- `csvTest/`: CSV file upload testing
- Each has `index.html` + `js/{App.jsx,Component.jsx}` + `api.js`

## Testing & Code Quality

**Testing**:
- Use `pytest` for Python tests in `test/` directory
- Example: `test/util/test_convert_json.py` shows testing patterns

**AI-Assisted Development**:
- Prompts in `.github/prompts/` for code review and fixes
- Review workflow: generate reviews in `{module}/review/`, then use fix prompts
- Python: `review_python.prompt.md` → `fix.prompt.md`
- React: `react_review.prompt.md` → `react_fix.prompt.md`

## Data Flow Patterns

**File Upload → Processing**:
1. Frontend sends FormData or base64 to Flask API
2. Flask validates with `validate_file_upload()`
3. Processing in respective modules (`llm/`, `machineLearning/`)
4. Results returned as JSON

**ML Inference**:
1. User input → `util.convert_json_to_model_input()`
2. Model loading from `param/` + scaler from `scaler/`
3. Prediction → response formatting

## Security Considerations

- CORS restricted to localhost origins
- File upload validation with size/type limits
- Secure filename handling with `werkzeug.secure_filename`
- API keys via environment variables (see `sample.env`)

## Common Patterns to Follow

- Use absolute imports from project root (enabled by `run_app.py`)
- Error handling with appropriate HTTP status codes
- Logging with `loguru.logger` instead of print statements
- Type annotations for all function parameters and returns
- File operations exclusively with `pathlib.Path`
