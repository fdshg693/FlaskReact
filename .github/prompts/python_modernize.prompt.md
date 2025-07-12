---
mode: agent
---

# Python Code Modernization Agent

## Objective
Analyze and modernize Python code in the current workspace to follow the latest best practices, improve code quality, and enhance maintainability while preserving functionality.

## Scope
- Target Python files (.py) in the current context
    - dont change other unrelated files
    - if other files should be changed because of the change already made, try to do the smallest change possible
    - Focus on the file in the current context
- Focus on code structure, syntax, and patterns
- Not necessarily appropriate for all projects, but useful for learning and code improvement purposes

## Modernization Guidelines

### 1. Code Structure & Organization
- Ensure proper module structure with `__init__.py` files
- Use appropriate package organization
- Implement clear separation of concerns
- Add proper docstrings following PEP 257

### 2. Type Hints & Annotations
- Add type hints to function parameters and return values
- Use modern typing constructs (e.g., `list[str]` instead of `List[str]` for Python 3.9+)
- Implement proper generic types where applicable
- Add `from __future__ import annotations` for forward compatibility

### 3. Modern Python Features
- Use f-strings instead of `.format()` or `%` formatting
- Implement dataclasses or Pydantic models for data structures
- Use pathlib instead of os.path for file operations
- Apply context managers for resource management
- Utilize match-case statements where appropriate (Python 3.10+)

### 4. Code Quality Improvements
- Remove unused imports and variables
- Simplify complex expressions and nested conditions
- Use list/dict comprehensions where readable
- Apply proper error handling with specific exception types
- Implement logging instead of print statements

### 5. Performance & Best Practices
- Use appropriate data structures (sets vs lists for membership testing)
- Implement lazy evaluation where beneficial
- Apply caching mechanisms for expensive operations
- Use generators for memory-efficient iterations

## Analysis Process
1. **File Discovery**: Identify all Python files in the workspace
2. **Code Review**: Analyze current code patterns and structure
3. **Modernization Plan**: Create a prioritized list of improvements
4. **Implementation**: Apply changes while preserving functionality
5. **Validation**: Ensure code still works after modifications

## Expected Outcomes
- Improved code readability and maintainability
- Better type safety and IDE support
- Enhanced performance where applicable
- Compliance with modern Python standards (PEP 8, PEP 484, etc.)
- Updated dependencies and import statements

## Constraints
- Preserve existing functionality and behavior
- Maintain backward compatibility where required
- Consider project-specific requirements and constraints
- Focus on changes that provide clear benefits

## Tools & Standards
- Follow PEP 8 style guidelines
- Use modern typing module features
- Apply Black or similar formatters
- Consider mypy for type checking
- Implement proper testing patterns