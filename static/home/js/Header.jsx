function Header() {
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
  };

  return (
    <header className="bg-white shadow-sm border-b mb-6">
      <div className="max-w-3xl mx-auto px-6 py-4 flex justify-between items-center">
        <h1 className="text-xl font-bold text-gray-800">画像・アイリス種予測システム</h1>
        
        <div className="flex items-center space-x-4">
          {user ? (
            <div className="flex items-center space-x-3">
              <span className="text-sm text-gray-600">
                ようこそ、{user.username}さん
              </span>
              <button
                onClick={handleLogout}
                className="bg-red-500 text-white px-3 py-1 rounded text-sm hover:bg-red-600 transition"
              >
                ログアウト
              </button>
            </div>
          ) : (
            <div className="text-sm text-gray-600">
              ログインしてください
            </div>
          )}
        </div>
      </div>
    </header>
  );
}