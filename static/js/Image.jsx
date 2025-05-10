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
    <div>
      <h1>画像を選択してね</h1>
      <input
        type="file"
        accept="image/*"
        onChange={handleChange}
      />
      {fileUrl && (
        <>
          <img src={fileUrl} alt="preview" width="100" />
          <div className="letter">
            判定された文字：{letter}
          </div>
        </>
      )}
    </div>
  );
}
