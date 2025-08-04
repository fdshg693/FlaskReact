// SQL AI Assistant Component - AI-related functionality
const { useState } = React;

function SqlAiAssistant({ currentQuestion, schema, query, onQuerySuggestion }) {
  const [aiQuestion, setAiQuestion] = useState('');
  const [aiResponse, setAiResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAiAssist = async () => {
    if (!aiQuestion.trim()) {
      setError('質問を入力してください');
      return;
    }

    setLoading(true);
    setError(null);
    setAiResponse(null);

    try {
      const schemaInfo = schema ? JSON.stringify(schema) : '';
      const response = await getAiSqlAssistance(aiQuestion, schemaInfo);
      setAiResponse(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestedQuery = () => {
    if (aiResponse && aiResponse.suggestedQuery) {
      onQuerySuggestion(aiResponse.suggestedQuery);
    }
  };

  const getQuestionHelp = async () => {
    if (!currentQuestion) {
      setError('現在の問題が見つかりません');
      return;
    }

    const helpQuestion = `次のSQL問題を解決するためのヒントを教えてください: ${currentQuestion.description || currentQuestion.question}`;
    setAiQuestion(helpQuestion);
    
    setLoading(true);
    setError(null);
    setAiResponse(null);

    try {
      const schemaInfo = schema ? JSON.stringify(schema) : '';
      const response = await getAiSqlAssistance(helpQuestion, schemaInfo);
      setAiResponse(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const explainQuery = async () => {
    if (!query.trim()) {
      setError('説明するクエリがありません');
      return;
    }

    const explainQuestion = `次のSQLクエリについて詳しく説明してください: ${query}`;
    setAiQuestion(explainQuestion);
    
    setLoading(true);
    setError(null);
    setAiResponse(null);

    try {
      const schemaInfo = schema ? JSON.stringify(schema) : '';
      const response = await getAiSqlAssistance(explainQuestion, schemaInfo);
      setAiResponse(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 mt-6">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
          <span className="text-white text-sm font-bold">AI</span>
        </div>
        <h2 className="text-xl font-semibold text-gray-800">AI アシスタント</h2>
      </div>

      {/* Quick Actions */}
      <div className="flex flex-wrap gap-2 mb-4">
        <button
          onClick={getQuestionHelp}
          disabled={loading || !currentQuestion}
          className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          問題のヒント
        </button>
        <button
          onClick={explainQuery}
          disabled={loading || !query.trim()}
          className="px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          クエリ説明
        </button>
      </div>

      {/* AI Question Input */}
      <div className="mb-4">
        <label htmlFor="ai-question" className="block text-sm font-medium text-gray-700 mb-2">
          AI に質問する
        </label>
        <div className="flex gap-2">
          <textarea
            id="ai-question"
            value={aiQuestion}
            onChange={(e) => setAiQuestion(e.target.value)}
            placeholder="SQLやデータベースについて何でも質問してください..."
            className="flex-1 border border-gray-300 rounded-lg p-3 text-sm focus:ring-2 focus:ring-purple-500 focus:border-purple-500 resize-none"
            rows="3"
          />
          <button
            onClick={handleAiAssist}
            disabled={loading || !aiQuestion.trim()}
            className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                処理中...
              </>
            ) : (
              '質問'
            )}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-4">
          <h4 className="text-red-800 font-medium">エラー</h4>
          <p className="text-red-600 mt-1 text-sm">{error}</p>
        </div>
      )}

      {/* AI Response */}
      {aiResponse && (
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-lg p-4">
          <h4 className="text-purple-800 font-medium mb-3">AI からの回答</h4>
          
          {/* Response Text */}
          {aiResponse.response && (
            <div className="mb-4">
              <div className="text-gray-800 whitespace-pre-wrap leading-relaxed">
                {aiResponse.response}
              </div>
            </div>
          )}

          {/* Suggested Query */}
          {aiResponse.suggestedQuery && (
            <div className="border-t border-purple-200 pt-4">
              <div className="flex justify-between items-start mb-2">
                <h5 className="text-purple-700 font-medium">提案されたクエリ</h5>
                <button
                  onClick={handleSuggestedQuery}
                  className="px-3 py-1 text-sm bg-purple-500 text-white rounded hover:bg-purple-600"
                >
                  クエリを使用
                </button>
              </div>
              <div className="bg-white border border-purple-200 rounded p-3">
                <pre className="text-sm text-gray-800 font-mono whitespace-pre-wrap">
                  {aiResponse.suggestedQuery}
                </pre>
              </div>
            </div>
          )}

          {/* Additional Tips */}
          {aiResponse.tips && aiResponse.tips.length > 0 && (
            <div className="border-t border-purple-200 pt-4 mt-4">
              <h5 className="text-purple-700 font-medium mb-2">追加のヒント</h5>
              <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                {aiResponse.tips.map((tip, index) => (
                  <li key={index}>{tip}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Learning Resources */}
          {aiResponse.resources && aiResponse.resources.length > 0 && (
            <div className="border-t border-purple-200 pt-4 mt-4">
              <h5 className="text-purple-700 font-medium mb-2">学習リソース</h5>
              <ul className="space-y-1 text-sm">
                {aiResponse.resources.map((resource, index) => (
                  <li key={index}>
                    <a 
                      href={resource.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-purple-600 hover:text-purple-800 underline"
                    >
                      {resource.title}
                    </a>
                    {resource.description && (
                      <span className="text-gray-600 ml-2">- {resource.description}</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Help Text */}
      <div className="mt-4 text-xs text-gray-500">
        💡 ヒント: 「SELECT文の書き方」「JOINの使い方」「集計関数について」など、具体的な質問をすると詳しい回答が得られます。
      </div>
    </div>
  );
}