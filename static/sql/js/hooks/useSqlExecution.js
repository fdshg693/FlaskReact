// Custom hook for SQL execution logic
const { useState, useCallback } = React;

function useSqlExecution() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [executionHistory, setExecutionHistory] = useState([]);
  const [validationStatus, setValidationStatus] = useState(null);

  const executeQuery = useCallback(async (sqlQuery = query, database = 'sample') => {
    if (!sqlQuery.trim()) {
      setError('クエリが空です');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const result = await executeSqlQuery(sqlQuery, database);
      setResults(result);
      
      // Add to execution history
      const historyEntry = {
        query: sqlQuery,
        results: result,
        timestamp: new Date(),
        database: database
      };
      setExecutionHistory(prev => [historyEntry, ...prev.slice(0, 9)]); // Keep last 10 entries
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [query]);

  const validateQuery = useCallback(async (sqlQuery = query) => {
    if (!sqlQuery.trim()) {
      setValidationStatus(null);
      return;
    }

    try {
      const validation = await validateSqlQuery(sqlQuery);
      setValidationStatus(validation);
      return validation;
    } catch (err) {
      setValidationStatus({ 
        valid: false, 
        error: err.message 
      });
      return { valid: false, error: err.message };
    }
  }, [query]);

  const clearResults = useCallback(() => {
    setResults(null);
    setError(null);
    setValidationStatus(null);
  }, []);

  const clearHistory = useCallback(() => {
    setExecutionHistory([]);
  }, []);

  const loadFromHistory = useCallback((historyEntry) => {
    setQuery(historyEntry.query);
    setResults(historyEntry.results);
    setError(null);
  }, []);

  const formatResults = useCallback((data) => {
    if (!data) return null;
    
    return {
      columns: data.columns || [],
      rows: data.rows || [],
      rowCount: data.rowCount || 0,
      executionTime: data.executionTime || 0,
      affectedRows: data.affectedRows || 0
    };
  }, []);

  return {
    query,
    setQuery,
    results: formatResults(results),
    error,
    loading,
    executionHistory,
    validationStatus,
    executeQuery,
    validateQuery,
    clearResults,
    clearHistory,
    loadFromHistory
  };
}