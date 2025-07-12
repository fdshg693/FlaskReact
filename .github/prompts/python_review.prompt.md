---
mode: agent
---

# Python Code Review Instructions

## Objective
Conduct a comprehensive review of the provided Python code to identify areas for improvement, suggest corrections, and enhance overall code quality.

## Review Methodology
Analyze the code systematically across multiple dimensions, prioritizing both immediate improvements and long-term maintainability. Consider breaking changes when they provide significant benefits to code quality, performance, or security.

## Review Criteria

### 1. Code Quality
- **Syntax and Style**: Adherence to PEP 8 and Python best practices
- **Design Patterns**: Appropriate use of design patterns and architectural principles
- **Error Handling**: Proper exception handling and error management
- **Testing**: Test coverage and testability considerations

### 2. Readability and Documentation
- **Code Clarity**: Variable naming, function structure, and logical flow
- **Documentation**: Docstrings, comments, and inline documentation
- **Type Hints**: Usage of type annotations for better code understanding
- **Code Organization**: Module structure and import organization

### 3. Performance
- **Algorithm Efficiency**: Time and space complexity analysis
- **Resource Usage**: Memory management and resource optimization
- **Bottlenecks**: Identification of performance-critical sections
- **Scalability**: Code's ability to handle increased load or data

### 4. Security
- **Input Validation**: Proper sanitization and validation of user inputs
- **Dependency Security**: Analysis of third-party library vulnerabilities
- **Data Protection**: Secure handling of sensitive information
- **Authentication/Authorization**: Security controls implementation

### 5. Maintainability
- **Code Modularity**: Separation of concerns and component isolation
- **Coupling and Cohesion**: Appropriate relationships between components
- **Extensibility**: Ease of adding new features or modifications
- **Dependency Management**: External library usage and version control

## Review Guidelines

### Critical Assessment
- **Major Refactoring**: Recommend significant structural changes when they improve code quality substantially
- **Breaking Changes**: Suggest modifications that may break current functionality if they address fundamental design flaws
- **Alternative Approaches**: Propose different implementation strategies when applicable

### Severity Classification
- **Critical**: Security vulnerabilities, major performance issues
- **High**: Significant design flaws, maintainability concerns
- **Medium**: Code quality improvements, readability enhancements
- **Low**: Style consistency, minor optimizations

## Output Requirements

### Review Document Format
Create a comprehensive review document named `review/review_{original_filename}.md` in the same directory as the original file.
If there is already a review file, just overwrite it with the new review.

### Required Sections
1. **Executive Summary**: High-level assessment and key findings
2. **Detailed Analysis**: Section-by-section breakdown by review criteria
3. **Priority Recommendations**: Categorized by severity and impact
4. **Code Examples**: Before/after comparisons for significant changes
5. **Implementation Roadmap**: Suggested order of improvements
6. **Additional Resources**: Relevant documentation or learning materials

### Formatting Standards
- Use clear markdown formatting with appropriate headers
- Include code blocks with syntax highlighting
- Provide actionable recommendations with specific examples
- Link to relevant Python documentation or best practices when applicable

## Success Metrics
The review should result in:
- Actionable improvement recommendations
- Clear prioritization of changes
- Detailed explanations for suggested modifications
- Enhanced code quality, security, and maintainability