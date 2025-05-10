function Image() {
    const [file, setFile] = useState(null);
    const [letter, setLetter] = useState('');

    // ファイル選択時のハンドラ
    const handleChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setFile(URL.createObjectURL(selected));  // 画像プレビュー用
    // ランダムなローマ字１文字を生成（大文字 A–Z）
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    const rand = letters[Math.floor(Math.random() * letters.length)];
    setLetter(rand);
    };

    return (
    <div>
        <h1>画像を選択してね</h1>
        <input type="file" accept="image/*" onChange={handleChange} />
        {file && (
        <>
            <img src={file} alt="preview" className="preview" width="100" />
            <div className="letter">{letter}</div>
        </>
        )}
    </div>
    )
}