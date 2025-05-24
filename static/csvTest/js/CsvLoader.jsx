const { useState } = React;

function CsvLoader() {
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
            //ユーザデータを学習する
            await trainUserData(results);
            setData(results.data);
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
            //ユーザデータを学習する
            await trainUserData(results);
            setData(results.data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-6 bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
            <div className="mb-4">
                <label className="block text-base font-medium mb-2">
                    ファイルを選択: &nbsp;
                    <input type="file" accept=".csv" onChange={handleFile} className="block mt-2 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 focus:outline-none focus:ring-2 focus:ring-blue-400 transition" />
                </label>
            </div>
            <div className="mb-4">
                <textarea
                    rows={6}
                    cols={60}
                    placeholder="CSV テキストをここに入力..."
                    value={csvText}
                    onChange={(e) => setCsvText(e.target.value)}
                    className="w-full border border-gray-300 rounded-lg p-3 mb-2 focus:outline-none focus:ring-2 focus:ring-blue-400 shadow-sm transition placeholder-gray-400 bg-white hover:border-blue-400"
                />
                <br />
                <button onClick={handleParseText} className="bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600 text-white font-semibold py-2 px-6 rounded-lg shadow-md transition transform hover:-translate-y-0.5 focus:outline-none focus:ring-2 focus:ring-blue-400">
                    テキストから解析
                </button>
            </div>
            <p className="font-semibold mb-2 text-lg text-indigo-700">学習したデータ</p>

            {loading && <p className="text-blue-500 animate-pulse">読み込み中...</p>}
            {error && <p className="text-red-500">エラー: {error}</p>}
            {data.length > 0 && (
                <div className="overflow-x-auto rounded-lg shadow-lg bg-white">
                    <table className="min-w-full border border-gray-300 rounded-lg overflow-hidden text-sm">
                        <thead className="bg-gradient-to-r from-indigo-100 to-blue-100 sticky top-0 z-10">
                            <tr>
                                {Object.keys(data[0]).map((h, i) => (
                                    <th
                                        key={i}
                                        className="px-4 py-3 border-b border-gray-300 text-left font-semibold text-indigo-700 tracking-wide uppercase sticky top-0 bg-gradient-to-r from-indigo-100 to-blue-100"
                                        style={{ border: '1px solid #cbd5e1' }}
                                    >
                                        {h}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {data.map((row, idx) => (
                                <tr
                                    key={idx}
                                    className={
                                        (idx % 2 === 0
                                            ? "bg-white hover:bg-blue-50"
                                            : "bg-blue-50 hover:bg-blue-100") +
                                        " transition"
                                    }
                                >
                                    {Object.keys(row).map((h, j) => (
                                        <td
                                            key={j}
                                            className="px-4 py-2 border-b border-gray-200 text-gray-700 group-hover:bg-blue-100 transition"
                                            style={{ border: '1px solid #cbd5e1' }}
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
root.render(<App />);