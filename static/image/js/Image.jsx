function Image() {
  const [fileUrl, setFileUrl] = useState(null);
  const [description, setDescription] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);

  // fileUrl が変わったら画像認識を呼び出す
  useEffect(() => {
    if (!fileUrl) return;

    (async () => {
      try {
        const result = await analyzeImage(selectedFile);
        setDescription(result);
      } catch (e) {
        setDescription('取得失敗');
      }
    })();
  }, [fileUrl]);

  // ファイル選択時のハンドラ
  const handleChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    // プレビュー用 URL を state に保存
    setFileUrl(URL.createObjectURL(file));
    // description は次の useEffect でセットされる
    setDescription('…認識中…');
  };

  return (
    <div className="max-w-md mx-auto p-6 bg-blue-300 rounded-2xl shadow-md space-y-4">
      <h3 className="text-2xl font-bold mb-4">画像判定</h3>
      <p className="text-2xl font-bold mb-4">判定したい画像をアップロードしてください</p>
      <input
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="mb-4"
      />
      {fileUrl && (
        <>
          <img src={fileUrl} alt="preview" width="100" className="mb-2 rounded shadow" />
          <div className="description text-lg text-gray-700">
            判定された画像：{description}
          </div>
        </>
      )}
    </div>
  );
}
