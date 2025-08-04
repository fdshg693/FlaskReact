// SQL Execution Panel Component - SQL input, execution & results display
const { useState } = React;

function SqlExecutionPanel({ 
  query, 
  setQuery, 
  results, 
  error, 
  loading, 
  executionHistory,
  validationStatus,
  onExecuteQuery,
  onValidateQuery,
  onClearResults,
  onClearHistory,
  onLoadFromHistory 
}) {
  const [showHistory, setShowHistory] = useState(false);

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
    // Auto-validate on change (debounced)
    setTimeout(() => {
      onValidateQuery(e.target.value);
    }, 500);
  };

  const handleKeyDown = (e) => {
    // Ctrl+Enter to execute query
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      onExecuteQuery();
    }
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      {/* SQL Input Panel */}
      <div className="xl:col-span-2 bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-gray-800">SQL クエリ</h2>
          <div className="flex gap-2">
            <button
              onClick={() => onValidateQuery()}
              disabled={loading || !query.trim()}
              className="px-3 py-1 text-sm bg-yellow-500 text-white rounded hover:bg-yellow-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              検証
            </button>
            <button
              onClick={() => onExecuteQuery()}
              disabled={loading || !query.trim()}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  実行中...
                </>
              ) : (
                <>
                  <span>実行</span>
                  <span className="text-xs">(Ctrl+Enter)</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Validation Status */}
        {validationStatus && (
          <div className={`mb-4 p-3 rounded-lg border ${
            validationStatus.valid 
              ? 'bg-green-50 border-green-200 text-green-800' 
              : 'bg-red-50 border-red-200 text-red-800'
          }`}>
            <div className="flex items-center gap-2">
              <span className="font-medium">
                {validationStatus.valid ? '✓ 構文正常' : '✗ 構文エラー'}
              </span>
              {validationStatus.error && (
                <span className="text-sm">: {validationStatus.error}</span>
              )}
            </div>
          </div>
        )}

        {/* SQL Textarea */}
        <textarea
          value={query}
          onChange={handleQueryChange}
          onKeyDown={handleKeyDown}
          placeholder="SELECT * FROM table_name;"
          className="w-full h-40 border border-gray-300 rounded-lg p-4 font-mono text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
        />

        <div className="mt-4 flex justify-between items-center text-sm text-gray-600">
          <div>
            Ctrl+Enter で実行 | 文字数: {query.length}
          </div>
          <button
            onClick={onClearResults}
            className="text-red-600 hover:text-red-800"
          >
            クリア
          </button>
        </div>

        {/* Results Panel */}
        <div className="mt-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-800">実行結果</h3>
            {results && (
              <div className="text-sm text-gray-600">
                {results.rowCount} 行 ({results.executionTime}ms)
              </div>
            )}
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <h4 className="text-red-800 font-medium">エラー</h4>
              <pre className="text-red-600 mt-2 text-sm whitespace-pre-wrap">{error}</pre>
            </div>
          )}

          {results && !error && (
            <div className="border border-gray-200 rounded-lg overflow-hidden">
              {results.columns && results.columns.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50">
                      <tr>
                        {results.columns.map((column, index) => (
                          <th key={index} className="px-4 py-3 text-left font-medium text-gray-700 border-b border-gray-200">
                            {column}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.rows && results.rows.map((row, rowIndex) => (
                        <tr key={rowIndex} className="border-b border-gray-200 hover:bg-gray-50">
                          {results.columns.map((column, colIndex) => (
                            <td key={colIndex} className="px-4 py-3 text-gray-700">
                              {row[column] !== null && row[column] !== undefined ? String(row[column]) : '-'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="p-4 text-center text-gray-500">
                  {results.affectedRows !== undefined 
                    ? `${results.affectedRows} 行が影響を受けました`
                    : 'クエリが正常に実行されました'
                  }
                </div>
              )}
            </div>
          )}

          {!results && !error && !loading && (
            <div className="text-center text-gray-500 py-8 border border-gray-200 rounded-lg">
              SQLクエリを入力して実行ボタンを押してください
            </div>
          )}
        </div>
      </div>

      {/* History Panel */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-800">実行履歴</h3>
          <div className="flex gap-2">
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              {showHistory ? '隠す' : '表示'}
            </button>
            {executionHistory.length > 0 && (
              <button
                onClick={onClearHistory}
                className="text-sm text-red-600 hover:text-red-800"
              >
                クリア
              </button>
            )}
          </div>
        </div>

        {showHistory && (
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {executionHistory.length === 0 ? (
              <p className="text-gray-500 text-sm text-center py-4">
                実行履歴がありません
              </p>
            ) : (
              executionHistory.map((entry, index) => (
                <div 
                  key={index} 
                  className="border border-gray-200 rounded-lg p-3 hover:bg-gray-50 cursor-pointer"
                  onClick={() => onLoadFromHistory(entry)}
                >
                  <div className="text-xs text-gray-500 mb-1">
                    {entry.timestamp.toLocaleString()}
                  </div>
                  <div className="font-mono text-sm text-gray-700 truncate">
                    {entry.query}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {entry.results?.rowCount || 0} 行
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}