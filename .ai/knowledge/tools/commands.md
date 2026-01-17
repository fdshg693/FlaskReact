# Useful commands

## List of Useful python Scripts
- `.ai/scripts/choice_selector.py`: 
    - A script to prompt the user to select one option from a list of choices.
    - `uv run python .ai/scripts/choice_selector.py "Select an option:" "Option 1" "Option 2" "Option 3"`
- `.ai/scripts/http_request.py`: 
    - A script to make HTTP requests and return the response.This is useful for testing APIs.
    - Especially recommended for windows environments where curl is unstable.
    - `uv run python http_request.py GET http://localhost:3000/api/todos`
- `.ai/scripts/procman.py`: 
    - A script to manage background processes.
    - Can start, stop, and check the status of processes.
    - Always recomended when double or more processes are needed.
        - frontend/backend servers starting. 
        - server starting, and API testing.

## List of formatting/linting/type-checking commands
- `uv run ruff check . --fix`:
    - Run ruff linter and automatically fix issues.
- `uv run ruff format .`:
    - Format all Python files in the current directory and subdirectories using ruff.
- `uv run pyright .`:
    - Perform type checking on all Python files in the current directory and subdirectories using pyright.
- `uv run pre-commit run --all-files`:
    - Run all pre-commit hooks on all files in the repository.