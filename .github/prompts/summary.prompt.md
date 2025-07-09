---
mode: agent
---

# File Summary and Documentation Agent

## Objective
Analyze the provided file content and generate comprehensive documentation under `summary` directory in the same directory as the original file.
This includes adding summary comments, evaluating the filename, and enhancing code readability.

## Output
`summary/summary_{original_filename}.md` in the same directory as the original file.


## Core Tasks
1. **Content Analysis**: Examine the file structure, functions, classes, and overall purpose
2. **Summary Generation**: Create concise, informative summary comments
3. **Filename Evaluation**: Assess if the current filename appropriately describes the file's functionality
4. **Documentation Enhancement**: Improve code readability through clear documentation

## Implementation Guidelines

### Comment Syntax by File Type
| File Type | Comment Syntax | Example |
|-----------|----------------|---------|
| Python (.py) | `#` | `# This module handles user authentication` |
| JavaScript (.js, .jsx, .ts, .tsx) | `//` | `// React component for data visualization` |
| YAML (.yml, .yaml) | `#` | `# Configuration file for deployment settings` |
| Markdown (.md) | `<!-- -->` | `<!-- Documentation for API endpoints -->` |
| JSON (.json) | `//` (if supported) | `// Application configuration schema` |
| HTML (.html) | `<!-- -->` | `<!-- Main landing page template -->` |

### Summary Structure
1. **Primary Purpose**: One-line description of the file's main function
2. **Key Components**: Brief overview of major functions, classes, or sections
3. **Dependencies**: Notable imports or external requirements (if applicable)
4. **Usage Context**: When and how this file is typically used

### Filename Evaluation Criteria
- **Descriptiveness**: Does the name clearly indicate the file's purpose?
- **Conventions**: Does it follow standard naming conventions for the project/language?
- **Specificity**: Is it specific enough to distinguish from similar files?
- **Length**: Is it appropriately concise while remaining descriptive?

## Output Requirements
1. **Enhanced File**: Original file content with summary comments added at the top
2. **Filename Recommendation**: If current name is inadequate, suggest a more appropriate name
    - use snake_case for Python, kebab-case for JavaScript, and PascalCase for classes
3. **Rationale**: Brief explanation for any filename change suggestions

## Examples

### Example 1: Python Function
**Before (No summary):**
```python
def calculate_area(radius):
    return 3.14 * radius * radius
```

**After (With summary):**
```python
# Mathematical utility function for geometric calculations
# Calculates the area of a circle using the formula π * r²
# Dependencies: None
# Usage: Called by geometry processing modules

def calculate_area(radius):
    return 3.14 * radius * radius
```
**Filename suggestion**: `circle_area_calculator.py` (if current name is generic like `utils.py`)

### Example 2: React Component
**Before:**
```javascript
export default function UserList({ users }) {
    return (
        <div>
            {users.map(user => <div key={user.id}>{user.name}</div>)}
        </div>
    );
}
```

**After:**
```javascript
// React component for displaying user information
// Renders a list of users with basic information display
// Props: users (array) - collection of user objects with id and name
// Usage: User management dashboard and admin panels

export default function UserList({ users }) {
    return (
        <div>
            {users.map(user => <div key={user.id}>{user.name}</div>)}
        </div>
    );
}
```

## Quality Standards
- Summary must be accurate and reflect actual file content
- Comments should be professional and concise
- Avoid redundant or obvious statements
- Focus on functionality and purpose rather than implementation details
- Ensure suggested filenames follow project conventions