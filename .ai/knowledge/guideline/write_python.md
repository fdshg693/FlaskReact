# Current Environment
- You are in a Python environment with the following libraries shown in requirements.txt
- you are using python3.13.5
- you are using uv 0.7
- you are under virtual environment (`.venv` directory)
- You are using a modern Python development environment with the following libraries:
  - `pathlib`
  - `mypy`
  - `ruff`
  - `pytest`
  - `loguru`
  - `pydantic`

# Modern Python Development Guidelines

## Core Requirements

### Python Version
- Follow PEP 8 style guidelines

### Modern Python Features (Required)
- **Type hints**: Add type annotations to all functions and variables
    - use mypy to check type hints
- **f-strings**: Use f-string formatting instead of `.format()` or `%` formatting
- **pathlib**: Use `pathlib.Path` for all file system operations instead of `os.path`
- **pydoc**: Use `pydoc` for generating documentation
- **loguru**: Use `loguru` for logging instead of the built-in `logging` module
- **ruff**: Use `ruff` for linting and formatting code
- **pytest**: Use `pytest` for testing if testing is important
- **pydantic**: Use `pydantic` for data validation and settings management

### Code Quality Standards
- Remove unnecessary imports
- Avoid deprecated libraries and methods
- Use modern Python idioms and best practices
- Ensure code is clean, readable, and maintainable
- use black for formatting  
- use pytest for testing if testing is important