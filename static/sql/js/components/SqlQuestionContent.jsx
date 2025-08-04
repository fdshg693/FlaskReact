// SQL Question Content Component - Problem display & database table visualization
function SqlQuestionContent({ currentQuestion, schema, loading, error }) {
  
  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <div className="flex items-center justify-center h-32">
          <div className="text-gray-500">問題を読み込み中...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-red-800 font-medium">エラーが発生しました</h3>
          <p className="text-red-600 mt-1">{error}</p>
        </div>
      </div>
    );
  }

  if (!currentQuestion) {
    return (
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <div className="text-center text-gray-500 py-8">
          問題が見つかりません。難易度を選択して「次の問題」をクリックしてください。
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
      {/* Question Panel */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          問題 {currentQuestion.id || '1'}
        </h2>
        
        <div className="space-y-4">
          {/* Question Title */}
          {currentQuestion.title && (
            <div>
              <h3 className="font-medium text-gray-700 mb-2">タイトル</h3>
              <p className="text-gray-800">{currentQuestion.title}</p>
            </div>
          )}

          {/* Question Description */}
          <div>
            <h3 className="font-medium text-gray-700 mb-2">問題文</h3>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-gray-800 whitespace-pre-wrap">
                {currentQuestion.description || currentQuestion.question || 
                 'SELECT文を使って、サンプルデータベースからデータを取得してください。'}
              </p>
            </div>
          </div>

          {/* Expected Output */}
          {currentQuestion.expectedOutput && (
            <div>
              <h3 className="font-medium text-gray-700 mb-2">期待される結果</h3>
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                <pre className="text-sm text-gray-700 whitespace-pre-wrap">
                  {currentQuestion.expectedOutput}
                </pre>
              </div>
            </div>
          )}

          {/* Hints */}
          {currentQuestion.hints && currentQuestion.hints.length > 0 && (
            <div>
              <h3 className="font-medium text-gray-700 mb-2">ヒント</h3>
              <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                {currentQuestion.hints.map((hint, index) => (
                  <li key={index}>{hint}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Database Schema Panel */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          データベーススキーマ
        </h2>
        
        {schema ? (
          <div className="space-y-4">
            {schema.tables && schema.tables.map((table, tableIndex) => (
              <div key={tableIndex} className="border border-gray-200 rounded-lg overflow-hidden">
                <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
                  <h3 className="font-medium text-gray-800">{table.name}</h3>
                  {table.description && (
                    <p className="text-sm text-gray-600 mt-1">{table.description}</p>
                  )}
                </div>
                
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="px-3 py-2 text-left font-medium text-gray-700">列名</th>
                        <th className="px-3 py-2 text-left font-medium text-gray-700">型</th>
                        <th className="px-3 py-2 text-left font-medium text-gray-700">制約</th>
                      </tr>
                    </thead>
                    <tbody>
                      {table.columns && table.columns.map((column, colIndex) => (
                        <tr key={colIndex} className="border-t border-gray-200">
                          <td className="px-3 py-2 font-mono text-blue-600">{column.name}</td>
                          <td className="px-3 py-2 text-gray-700">{column.type}</td>
                          <td className="px-3 py-2 text-gray-600">
                            {column.constraints ? column.constraints.join(', ') : '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Sample Data */}
                {table.sampleData && table.sampleData.length > 0 && (
                  <div className="border-t border-gray-200">
                    <div className="bg-gray-50 px-4 py-2">
                      <h4 className="text-sm font-medium text-gray-700">サンプルデータ</h4>
                    </div>
                    <div className="overflow-x-auto max-h-40 overflow-y-auto">
                      <table className="w-full text-sm">
                        <thead className="bg-gray-100 sticky top-0">
                          <tr>
                            {table.columns && table.columns.map((column, colIndex) => (
                              <th key={colIndex} className="px-3 py-2 text-left font-medium text-gray-700">
                                {column.name}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {table.sampleData.slice(0, 5).map((row, rowIndex) => (
                            <tr key={rowIndex} className="border-t border-gray-200">
                              {table.columns && table.columns.map((column, colIndex) => (
                                <td key={colIndex} className="px-3 py-2 text-gray-700">
                                  {row[column.name] || '-'}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-gray-500 text-center py-8">
            スキーマ情報を読み込み中...
          </div>
        )}
      </div>
    </div>
  );
}