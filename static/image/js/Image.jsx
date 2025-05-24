function Image() {
  const [fileUrl, setFileUrl] = useState(null);
  const [letter, setLetter] = useState('');

  // fileUrl が変わったら文字認識を呼び出す
  useEffect(() => {
    if (!fileUrl) return;

    (async () => {
      try {
        // fetchImage は fileUrl からサーバーに投げるなど適宜実装
        const result = await fetchImage(fileUrl);
        console.log(result);
        setLetter(result);
      } catch (e) {
        setLetter('取得失敗');
      }
    })();
  }, [fileUrl]);

  // ファイル選択時のハンドラ
  const handleChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    // プレビュー用 URL を state に保存
    setFileUrl(URL.createObjectURL(selected));
    // letter は次の useEffect でセットされる
    setLetter('…認識中…');
  };

  return (
    <div className="max-w-md mx-auto p-6 bg-blue-300 rounded-2xl shadow-md space-y-4">
      <h1 className="text-2xl font-bold mb-4">判定したい画像をアップロードしてください</h1>
      <input
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="mb-4"
      />
      {fileUrl && (
        <>
          <img src={fileUrl} alt="preview" width="100" className="mb-2 rounded shadow" />
          <div className="letter text-lg text-gray-700">
            判定された文字：{letter}
          </div>
        </>
      )}
    </div>
  );
}
