// Main SQL App Component - Coordinates all sub-components
const { useState, useEffect } = React;

function App() {
  // Use custom hooks for state management
  const questionState = useSqlQuestionState();
  const executionState = useSqlExecution();

  const handleQuerySuggestion = (suggestedQuery) => {
    executionState.setQuery(suggestedQuery);
  };

  // Auto-load a sample query when a question changes
  useEffect(() => {
    if (questionState.currentQuestion && questionState.currentQuestion.sampleQuery && !executionState.query) {
      executionState.setQuery(questionState.currentQuestion.sampleQuery);
    }
  }, [questionState.currentQuestion]);

  return (
    <div className="font-sans">
      {/* Header with navigation controls */}
      <SqlQuestionHeader
        difficulty={questionState.difficulty}
        setDifficulty={questionState.setDifficulty}
        onNextQuestion={questionState.nextQuestion}
        onPreviousQuestion={questionState.previousQuestion}
        onResetQuestions={questionState.resetQuestions}
        questionHistory={questionState.questionHistory}
        loading={questionState.loading}
      />

      {/* Question content and database schema */}
      <SqlQuestionContent
        currentQuestion={questionState.currentQuestion}
        schema={questionState.schema}
        loading={questionState.loading}
        error={questionState.error}
      />

      {/* SQL execution panel */}
      <SqlExecutionPanel
        query={executionState.query}
        setQuery={executionState.setQuery}
        results={executionState.results}
        error={executionState.error}
        loading={executionState.loading}
        executionHistory={executionState.executionHistory}
        validationStatus={executionState.validationStatus}
        onExecuteQuery={executionState.executeQuery}
        onValidateQuery={executionState.validateQuery}
        onClearResults={executionState.clearResults}
        onClearHistory={executionState.clearHistory}
        onLoadFromHistory={executionState.loadFromHistory}
      />

      {/* AI Assistant */}
      <SqlAiAssistant
        currentQuestion={questionState.currentQuestion}
        schema={questionState.schema}
        query={executionState.query}
        onQuerySuggestion={handleQuerySuggestion}
      />
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);