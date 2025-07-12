const { useState } = React;

function MultipleIris() {
    const [data, setData] = useState([]);
    const [error, setError] = useState(null);
    const [csvText, setCsvText] = useState('');
    const [loading, setLoading] = useState(false);

    // ファイルから読み込む
    const handleFile = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        setLoading(true);
        setError(null);
        try {
            const results = await new Promise((resolve, reject) => {
                Papa.parse(file, {
                    header: true,
                    skipEmptyLines: true,
                    complete: (results) => resolve(results),
                    error: (err) => reject(err)
                });
            });
            //　複数のユーザ入力を判定する
            const result = await fetchMultipleIrisSpecies(results);
            setData(result);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // テキスト入力から読み込む
    const handleParseText = async () => {
        if (!csvText) return;
        setLoading(true);
        setError(null);
        try {
            const results = await new Promise((resolve, reject) => {
                try {
                    const parsed = Papa.parse(csvText, { header: true, skipEmptyLines: true });
                    resolve(parsed);
                } catch (err) {
                    reject(err);
                }
            });
            //　複数のユーザ入力を判定する
            const result = await fetchMultipleIrisSpecies(results);
            setData(result);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-6">
            <p>アイリスcsvのデータをcsvから読み込む（ヘッダ必須・順不同です）</p>
            <div className="mb-4">
                <label className="block text-base font-medium mb-2">
                    アイリス予測用CSVデータ: &nbsp;
                    <input type="file" accept=".csv" onChange={handleFile} className="block mt-2" />
                </label>
            </div>
            <div className="mb-4">
                <textarea
                    rows={6}
                    cols={60}
                    placeholder="CSV テキストをここに入力...（ヘッダ必須・順不同です）"
                    value={csvText}
                    onChange={(e) => setCsvText(e.target.value)}
                    className="w-full border border-gray-300 rounded p-2 mb-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
                <br />
                <button onClick={handleParseText} className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded shadow">
                    テキストcsvから予測
                </button>
            </div>
            <p className="font-semibold mb-2">入力したデータ</p>

            {loading && <p className="text-blue-500">計算中</p>}
            {error && <p className="text-red-500">エラー: {error}</p>}
            {data.length > 0 && (
                <div className="overflow-x-auto">
                    <table className="min-w-full border border-gray-500 rounded-lg overflow-hidden" style={{ borderCollapse: 'collapse' }}>
                        <thead className="bg-gray-100">
                            <tr>
                                {Object.keys(data[0]).map((h, i) => (
                                    <th
                                        key={i}
                                        className="px-4 py-2 border border-gray-500 text-left"
                                        style={{ border: '1px solid #6b7280' }}
                                    >
                                        {h}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {data.map((row, idx) => (
                                <tr key={idx} className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                                    {Object.keys(row).map((h, j) => (
                                        <td
                                            key={j}
                                            className="px-4 py-2 border border-gray-500"
                                            style={{ border: '1px solid #6b7280' }}
                                        >
                                            {row[h]}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<MultipleIris />);