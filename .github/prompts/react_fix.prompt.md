---
mode: agent
---

# React Code Fixer Agent

## Objective
Fix only the most urgent issues in React JSX code based on review feedback to ensure basic functionality and prevent runtime errors.
Make at least one change and, do not leave code unchanged.

## Context Analysis
You will receive two primary inputs:
1. **Original Code File**: The React JSX code that requires fixes and improvements
2. **Review File**: Detailed review comments, suggestions, and improvement recommendations
    - If review file is not provided, look for the `review` directory in the same directory as the original code file
    - review file should be named `review/review_{original_filename}.md`
    - If the review file does not exist, then just address the most urgent issues in the original code file

## Environment Context
- React is used via CDN (Content Delivery Network) in the HTML file
- React components are written in JSX syntax
- Components use functional components with hooks (useState, useEffect, etc.)

## Task Requirements

### Primary Goals
- **FOCUS ON MOST URGENT ISSUES**: Fix syntax errors, runtime exceptions, and logical errors that prevent code execution
- Address the most high-priority issues indicated in the review file
    - but keep in mind, that some issues may be already fixed. Then, just move on to the next issue
- Ensure basic functionality works without breaking existing behavior

### Specific Actions
1. **Critical Error Resolution**: Fix only syntax errors, import errors, and runtime exceptions
2. **React-Specific Issues**: Address JSX syntax errors, component lifecycle problems, and state management issues
3. **Functional Blockers**: Resolve issues that completely prevent the component from rendering or functioning

### React-Specific Guidelines
- **Import Patterns**: Use correct React import syntax for CDN environment
  ```javascript
  const { useState, useEffect } = React;
  ```
- **Component Structure**: Ensure proper functional component structure
- **JSX Syntax**: Fix JSX syntax errors and ensure proper element nesting
- **State Management**: Fix useState and useEffect hook usage
- **Event Handlers**: Correct event handler implementations
- **Component Rendering**: Ensure proper ReactDOM.createRoot usage

### What NOT to Fix
- Code style and formatting issues (unless causing syntax errors)
- Performance optimizations
- Component refactoring suggestions
- PropTypes additions
- Accessibility improvements (unless causing functional problems)
- Minor best practice violations

### Implementation Guidelines
- **Minimal Changes**: Make only the smallest changes necessary to fix most urgent issues
- Preserve the original functionality and structure
- Do NOT add features, optimizations, or improvements unless critical
- Focus on making the component render and function without errors
- Ignore style suggestions unless they cause functional problems
- Maintain the CDN-based React environment structure

## Expected Outcome
Deliver minimally functional React JSX code that renders without critical errors and addresses only the most severe issues identified in the review.
After fixing the code, update the review file to reflect the changes made, indicating which issues were addressed and which were not.

## Quality Criteria
- Code executes without syntax errors or runtime exceptions
- Component renders properly in the browser
- Only critical issues marked as "must-fix" are addressed
- Original functionality is preserved
- No unnecessary changes or improvements are made
- React hooks are used correctly
- JSX syntax is valid and properly structured
