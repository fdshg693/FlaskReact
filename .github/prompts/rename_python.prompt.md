---
mode: agent
---

# Python Code Refactoring: Rename Functions and Variables

Refactor the Python code to use more descriptive names that follow Python naming conventions (PEP 8).
And suggest new filenames if the current names do not accurately reflect the file's purpose or content.

## Naming Rules

### Functions and Variables
- Use `snake_case` for function names and variable names
- Use descriptive names that clearly indicate the purpose or content
- Avoid single-letter variables except for short loops (i, j, k) or mathematical operations (x, y, z)
- Avoid abbreviations unless they are well-established in the domain (e.g., `url`, `html`, `json`, `api`)

### Classes
- Use `PascalCase` for class names
- Use descriptive nouns that represent the entity or concept

### Constants
- Use `UPPER_SNAKE_CASE` for constants
- Group related constants together

### Private Members
- Prefix with single underscore `_` for internal use
- Prefix with double underscore `__` for name mangling (rare cases)

### File Name
- Use `snake_case` for Python files
- Ensure the filename reflects the file's purpose or content

## Examples

### Before (Poor naming):
```python
def calc(x, y):
    temp = x * 2
    res = temp + y
    return res

data = [1, 2, 3]
df = load_data()
```

### After (Improved naming):
```python
def calculate_double_plus_value(base_number, additional_value):
    doubled_base = base_number * 2
    final_result = doubled_base + additional_value
    return final_result

temperature_readings = [1, 2, 3]
user_dataframe = load_data()
```

## Guidelines
- Consider the context and domain when choosing names
- Prefer clarity over brevity
- Use verbs for functions that perform actions
- Use nouns for variables that store data
- Be consistent with naming patterns throughout the codebase
- Update related documentation and comments when renaming