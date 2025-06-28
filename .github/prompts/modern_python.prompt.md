---
mode: agent
---
# CUrrent Environment
- You are in a Python environment with the following libraries shown in requirements.txt
- you are using python3.13.3
- you are using pip 
- you are under virtual environment
- You are using a modern Python development environment with the following libraries:
  - `pathlib`
  - `mypy`
  - `black`
  - `pytest`

# Modern Python Development Guidelines

## Core Requirements

### Python Version
- Follow PEP 8 style guidelines

### Modern Python Features (Required)
- **Type hints**: Add type annotations to all functions and variables
    - use mypy to check type hints
- **f-strings**: Use f-string formatting instead of `.format()` or `%` formatting
- **pathlib**: Use `pathlib.Path` for all file system operations instead of `os.path`

### Code Quality Standards
- Remove unnecessary imports
- Avoid deprecated libraries and methods
- Use modern Python idioms and best practices
- Ensure code is clean, readable, and maintainable
- use black for formatting  
- use pytest for testing if testing is important