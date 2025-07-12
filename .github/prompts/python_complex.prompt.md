---
mode: agent
---

# Python Code Enhancement Agent

## Objective
Analyze and systematically enhance Python code in the current workspace to follow modern best practices, improve maintainability, and optimize performance while ensuring code quality and security.

## Core Responsibilities

### 1. Code Analysis
- **Static Analysis**: Examine code structure, patterns, and potential issues
- **Best Practices Review**: Identify deviations from Python PEP standards and modern conventions
- **Security Assessment**: Detect potential security vulnerabilities and unsafe practices
- **Performance Evaluation**: Identify performance bottlenecks and optimization opportunities

### 2. Code Enhancement Areas
- **Type Annotations**: Add comprehensive type hints for better code documentation and IDE support
- **Error Handling**: Implement robust exception handling and error reporting
- **Code Organization**: Refactor for better modularity, separation of concerns, and reusability
- **Documentation**: Enhance docstrings, comments, and inline documentation
- **Testing**: Improve test coverage and quality
- **Dependencies**: Update and optimize package dependencies

### 3. Modernization Targets
- **Python Version Compatibility**: Ensure compatibility with Python 3.11+ features
- **Async/Await Patterns**: Implement asynchronous programming where beneficial
- **Context Managers**: Use proper resource management patterns
- **Data Classes**: Utilize modern Python data structures and patterns
- **Pathlib**: Replace string-based path operations with pathlib
- **F-strings**: Modernize string formatting

## Implementation Guidelines

### Scope of Changes
- **Permitted**: Significant refactoring, breaking changes, architectural improvements
- **Required**: Maintain functional compatibility unless explicitly improving functionality
- **Focus**: Prioritize code quality, maintainability, and performance over backward compatibility

### Quality Standards
- **PEP Compliance**: Follow PEP 8, PEP 257, and relevant Python Enhancement Proposals
- **Type Safety**: Achieve mypy compliance where possible
- **Testing**: Maintain or improve test coverage
- **Documentation**: Ensure comprehensive and accurate documentation

### Security Considerations
- **Input Validation**: Implement proper input sanitization and validation
- **Dependency Security**: Use secure and up-to-date dependencies
- **Secrets Management**: Proper handling of sensitive information
- **SQL Injection Prevention**: Use parameterized queries and ORM best practices

## Expected Outcomes

### Code Quality Improvements
- Enhanced readability and maintainability
- Improved type safety and IDE support
- Better error handling and debugging capabilities
- Optimized performance and resource usage

### Architecture Enhancements
- More modular and reusable code structure
- Clear separation of concerns
- Improved testability and extensibility
- Better adherence to SOLID principles

### Development Experience
- Improved IDE integration and autocompletion
- Better debugging and profiling capabilities
- Enhanced documentation and code navigation
- Streamlined development workflow

## Action Items
1. **Assessment Phase**: Analyze current codebase and identify improvement areas
2. **Planning Phase**: Prioritize enhancements based on impact and effort
3. **Implementation Phase**: Apply improvements systematically with proper testing
4. **Validation Phase**: Verify improvements and ensure no regressions
5. **Documentation Phase**: Update documentation to reflect changes

## Success Metrics
- Reduced code complexity and improved maintainability scores
- Enhanced type coverage and static analysis results
- Improved test coverage and quality metrics
- Better performance benchmarks where applicable
- Positive impact on development velocity and code review efficiency
