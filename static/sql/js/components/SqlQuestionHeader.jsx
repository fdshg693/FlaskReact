// SQL Question Header Component - Navigation & header controls
const { useState } = React;

function SqlQuestionHeader({ 
  difficulty, 
  setDifficulty, 
  onNextQuestion, 
  onPreviousQuestion, 
  onResetQuestions,
  questionHistory,
  loading 
}) {
  const difficulties = [
    { value: 'beginner', label: '初級', color: 'green' },
    { value: 'intermediate', label: '中級', color: 'yellow' },
    { value: 'advanced', label: '上級', color: 'red' }
  ];

  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        {/* Title */}
        <div>
          <h1 className="text-2xl font-bold text-gray-800">SQL クエリ実行・学習</h1>
          <p className="text-gray-600 mt-1">データベースクエリの練習と実行環境</p>
        </div>

        {/* Controls */}
        <div className="flex flex-col sm:flex-row gap-3">
          {/* Difficulty Selector */}
          <div className="flex items-center gap-2">
            <label htmlFor="difficulty" className="text-sm font-medium text-gray-700">
              難易度:
            </label>
            <select
              id="difficulty"
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-1 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              disabled={loading}
            >
              {difficulties.map(diff => (
                <option key={diff.value} value={diff.value}>
                  {diff.label}
                </option>
              ))}
            </select>
          </div>

          {/* Navigation Buttons */}
          <div className="flex gap-2">
            <button
              onClick={onPreviousQuestion}
              disabled={questionHistory.length === 0 || loading}
              className="px-3 py-1 text-sm bg-gray-500 text-white rounded hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              ← 前の問題
            </button>
            
            <button
              onClick={onNextQuestion}
              disabled={loading}
              className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? '読み込み中...' : '次の問題 →'}
            </button>
            
            <button
              onClick={onResetQuestions}
              disabled={loading}
              className="px-3 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              リセット
            </button>
          </div>
        </div>
      </div>

      {/* Question Progress */}
      {questionHistory.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <span>問題履歴:</span>
            <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded">
              {questionHistory.length} 問完了
            </span>
          </div>
        </div>
      )}
    </div>
  );
}