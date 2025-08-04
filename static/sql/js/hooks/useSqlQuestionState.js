// Custom hook for SQL question state management
const { useState, useEffect, useCallback } = React;

function useSqlQuestionState() {
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [difficulty, setDifficulty] = useState('beginner');
  const [questionHistory, setQuestionHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [schema, setSchema] = useState(null);

  // Load initial question when difficulty changes
  useEffect(() => {
    loadQuestion();
  }, [difficulty]);

  // Load database schema on mount
  useEffect(() => {
    loadSchema();
  }, []);

  const loadQuestion = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const questionsData = await getSqlQuestions(difficulty);
      if (questionsData.questions && questionsData.questions.length > 0) {
        setCurrentQuestion(questionsData.questions[0]);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [difficulty]);

  const loadSchema = useCallback(async () => {
    try {
      const schemaData = await getDatabaseSchema('sample');
      setSchema(schemaData);
    } catch (err) {
      console.error('Failed to load schema:', err);
    }
  }, []);

  const nextQuestion = useCallback(async () => {
    if (currentQuestion) {
      setQuestionHistory(prev => [...prev, currentQuestion]);
    }
    await loadQuestion();
  }, [currentQuestion, loadQuestion]);

  const previousQuestion = useCallback(() => {
    if (questionHistory.length > 0) {
      const previousQ = questionHistory[questionHistory.length - 1];
      setQuestionHistory(prev => prev.slice(0, -1));
      setCurrentQuestion(previousQ);
    }
  }, [questionHistory]);

  const resetQuestions = useCallback(() => {
    setQuestionHistory([]);
    setCurrentQuestion(null);
    loadQuestion();
  }, [loadQuestion]);

  return {
    currentQuestion,
    difficulty,
    setDifficulty,
    questionHistory,
    loading,
    error,
    schema,
    nextQuestion,
    previousQuestion,
    resetQuestions,
    loadQuestion
  };
}