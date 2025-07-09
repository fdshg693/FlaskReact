---
mode: agent
---

# Python Code Fixer Agent

## Objective
Fix only the most urgent issues in Python code based on review feedback to ensure basic functionality and prevent runtime errors.
Make at least one change and, do not leave code unchanged.

## Context Analysis
You will receive two primary inputs:
1. **Original Code File**: The Python code that requires fixes and improvements
2. **Review File**: Detailed review comments, suggestions, and improvement recommendations
    - If review file is not provided, look for the `review` directory in the same directory as the original code file
    - review file should be named `review/review_{original_filename}.md`
    - If the review file does not exist, then just address the most urgent issues in the original code file

## Task Requirements

### Primary Goals
- **FOCUS ON MOST URGENT ISSUES**: Fix syntax errors, runtime exceptions, and logical errors that prevent code execution
- Address the most high-priority issues indicated in the review file
    - but keep in mind, that some issues may be already fixed. Then, just move on to the next issue
- Ensure basic functionality works without breaking existing behavior

### Specific Actions
1. **Critical Error Resolution**: Fix only syntax errors, import errors, and runtime exceptions
2. **Security Vulnerabilities**: Address only severe security issues that pose immediate risks
3. **Functional Blockers**: Resolve issues that completely prevent the code from running

### What NOT to Fix
- Code style and formatting issues (unless causing syntax errors)
- Performance optimizations
- Documentation improvements
- Refactoring suggestions
- Type hints additions
- Minor best practice violations

### Implementation Guidelines
- **Minimal Changes**: Make only the smallest changes necessary to fix most urgent issues
- Preserve the original functionality and structure
- Do NOT add features, optimizations, or improvements unless critical
- Focus on making the code run without errors
- Ignore style suggestions unless they cause functional problems

## Expected Outcome
Deliver minimally functional Python code that executes without critical errors and addresses only the most severe issues identified in the review.
After fixing the code, update the review file to reflect the changes made, indicating which issues were addressed and which were not.

## Quality Criteria
- Code executes without syntax errors or runtime exceptions
- Only critical issues marked as "must-fix" are addressed
- Original functionality is preserved
- No unnecessary changes or improvements are made