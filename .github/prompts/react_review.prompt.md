---
mode: agent
---

# React CDN Code Review Instructions

## Objective
Conduct a comprehensive review of React code files within the current context to identify areas for improvement, suggest corrections, and enhance overall code quality.

## Scope
Review ONLY the React related files present in the current context.
You can read other related files if they are necessary for understanding the React code. 
But do not analyze files that are not related to React or outside the current context.
and review should be done to the files in the context only.

Target files include:
- `.jsx` files (React components)
- `.html` files (containing React CDN setup)
- `.js` files (API utilities and React-related JavaScript)

## Review Process

### 1. Preparation
- Create a `review` folder in the same directory as the code being reviewed
- Focus only on the React files identified in the current context
- Do not analyze files outside the specified scope

### 2. Code Analysis Areas
Evaluate the following aspects of the React code:

### 3. Output
create a review file named `review/review_{original_filename}.md` for each React file reviewed, where `{original_filename}` is the name of the file being reviewed.

#### Structure & Organization
- Component structure and hierarchy
- File organization and naming conventions
- Code modularity and reusability

#### React Best Practices
- Proper use of React hooks (useState, useEffect, etc.)
- Component lifecycle management
- State management patterns
- Props handling and validation

#### CDN-Specific Considerations
- Proper CDN library loading order
- Version compatibility between React and related libraries
- Performance implications of CDN usage
- Security considerations for external CDN resources

#### Code Quality
- Code readability and maintainability
- Error handling and edge cases
- Performance optimizations
- Accessibility considerations

#### Technical Implementation
- Proper JSX syntax and patterns
- Event handling implementation
- API integration patterns
- Data flow and state updates

### 3. Output Requirements
- Create a review file named `review/review_{original_filename}.md`
- Structure the review with clear sections and actionable feedback
- Include specific code examples where improvements are needed
- Provide concrete suggestions for enhancement

## Review Template Structure
```markdown
# Review for {filename}

## Summary
Brief overview of the code's purpose and overall assessment.

## Strengths
- List positive aspects of the code

## Areas for Improvement

### Critical Issues
- Major problems that need immediate attention

### Best Practice Violations
- React/JavaScript best practices not followed

### Performance Concerns
- Performance-related issues and optimizations

### Code Quality Issues
- Readability, maintainability, and structure improvements

## Specific Recommendations
1. Detailed actionable suggestions with code examples
2. Priority level (High/Medium/Low)
3. Implementation guidance

## Conclusion
Overall assessment and next steps.
```

## Success Criteria
- All identified issues in the workspace React files are clearly documented
- Suggestions are actionable and specific to the current codebase
- Review maintains focus on React CDN implementation patterns within the context
- Output follows the specified file naming convention
- Analysis is limited to files present in the current context only