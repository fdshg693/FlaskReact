const { useState, useEffect } = React;

function App() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">読み込み中...</p>
        </div>
      </div>
    );
  }

  return (
    <AuthProvider>
      <div className="min-h-screen bg-gray-50">
        <Header />
        
        <div className="max-w-3xl mx-auto p-6">
          {user ? (
            <div>
              {/* Navigation Menu */}
              <nav className="mb-12">
                <ul className="grid grid-cols-1 sm:grid-cols-4 gap-4">
                  <li>
                    <a href="/" className="block p-4 bg-white rounded-lg shadow hover:shadow-md transition">
                      <span className="font-semibold">トップ</span>
                      <p className="text-sm text-gray-500">ホームページのトップへ戻る</p>
                    </a>
                  </li>
                  <li>
                    <a href="/home" className="block p-4 bg-white rounded-lg shadow hover:shadow-md transition">
                      <span className="font-semibold">アイリス</span>
                      <p className="text-sm text-gray-500">アイリス種別判定</p>
                    </a>
                  </li>
                  <li>
                    <a href="/csvTest" className="block p-4 bg-white rounded-lg shadow hover:shadow-md transition">
                      <span className="font-semibold">CSV</span>
                      <p className="text-sm text-gray-500">CSVを読み込んで表示</p>
                    </a>
                  </li>
                  <li>
                    <a href="/image" className="block p-4 bg-white rounded-lg shadow hover:shadow-md transition">
                      <span className="font-semibold">画像判定</span>
                      <p className="text-sm text-gray-500">画像をアップロードして判定</p>
                    </a>
                  </li>
                </ul>
              </nav>

              {/* Main Content */}
              <div className="font-sans">
                <Iris />
                <MultipleIris />
              </div>
            </div>
          ) : (
            <Login onLoginSuccess={(userData) => {
              // This will be handled by the AuthContext
            }} />
          )}
        </div>
      </div>
    </AuthProvider>
  );
}

function AppWithAuth() {
  return (
    <AuthProvider>
      <App />
    </AuthProvider>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<AppWithAuth />);
