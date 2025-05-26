function TextSplit({ rawText, setRawText }) {
    const [splitText, setSplitText] = useState(['']);
    const [clicked, setClicked] = useState(false);

    useEffect(() => {
        if (!clicked) return;

        (async () => {
            try {
                const chunks = await splitTextAPI(rawText);
                setSplitText(chunks);
            } catch (e) {
                setSplitText(['取得失敗']);
            } finally {
                setClicked(false);
            }
        })();
    }, [clicked, rawText]);

    return (
        <div className="max-w-md mx-auto p-6 bg-blue-300 rounded-2xl shadow-md space-y-4">
            <h3 className="text-2xl font-bold mb-4">テキスト分割</h3>
            <p className="text-2xl font-bold mb-4">分割したいテキストを入力してください</p>
            <textarea
                rows="10"
                className="w-full p-2 border border-gray-300 rounded mb-4"
                placeholder="ここにテキストを入力してください"
                value={rawText}
                onChange={(e) => setRawText(e.target.value)}
            />
            <button onClick={() => setClicked(true)}>分割</button>
            {rawText && (
                <>
                    <div className="split-text text-lg text-gray-700 mt-4">
                        分割されたテキスト：<br />
                        {splitText.map((chunk, index) => (
                            <div key={index} className="mb-2 p-2 bg-white rounded shadow">
                                {chunk}
                            </div>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
}

export default TextSplit;
