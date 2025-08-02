import React from 'react';
import Papa from 'papaparse';
import { fetchMultipleIrisSpecies } from './api.js';

const { useState } = React;

export function MultipleIris() {
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

    return React.createElement('div', { className: "p-6" },
        React.createElement('p', null, 'アイリスcsvのデータをcsvから読み込む（ヘッダ必須・順不同です）'),
        React.createElement('div', { className: "mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg" },
            React.createElement('p', { className: "font-semibold text-blue-800 mb-2" }, '期待する入力形式：'),
            React.createElement('p', { className: "text-blue-700 mb-2" }, 'CSV ファイルは以下のヘッダを含む必要があります：'),
            React.createElement('code', { className: "block bg-blue-100 p-2 rounded text-sm text-blue-900 mb-2" },
                'sepal.length,sepal.width,petal.length,petal.width'
            ),
            React.createElement('pre', { className: "bg-blue-100 p-2 rounded text-sm text-blue-900" },
                React.createElement('code', { className: "whitespace-pre text-blue-700" },
`参考例：
sepal.length,sepal.width,petal.length,petal.width
5.1,3.5,1.4,0.2
4.9,3.0,1.4,0.2`
                )
            )
        ),
        React.createElement('div', { className: "mb-4" },
            React.createElement('label', { className: "block text-base font-medium mb-2" },
                'アイリス予測用CSVデータ: \u00A0',
                React.createElement('input', {
                    type: "file",
                    accept: ".csv",
                    onChange: handleFile,
                    className: "block mt-2"
                })
            )
        ),
        React.createElement('div', { className: "mb-4" },
            React.createElement('textarea', {
                rows: 6,
                cols: 60,
                placeholder: "CSV テキストをここに入力...（ヘッダ必須・順不同です）",
                value: csvText,
                onChange: (e) => setCsvText(e.target.value),
                className: "w-full border border-gray-300 rounded p-2 mb-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
            }),
            React.createElement('br'),
            React.createElement('button', {
                onClick: handleParseText,
                className: "bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded shadow"
            }, 'テキストcsvから予測')
        ),
        React.createElement('p', { className: "font-semibold mb-2" }, '入力したデータ'),

        loading && React.createElement('p', { className: "text-blue-500" }, '計算中'),
        error && React.createElement('p', { className: "text-red-500" }, `エラー: ${error}`),
        data.length > 0 && React.createElement('div', { className: "overflow-x-auto" },
            React.createElement('table', {
                className: "min-w-full border border-gray-500 rounded-lg overflow-hidden",
                style: { borderCollapse: 'collapse' }
            },
                React.createElement('tbody', null,
                    data.map((row, idx) =>
                        React.createElement('tr', {
                            key: idx,
                            className: idx % 2 === 0 ? "bg-white" : "bg-gray-50"
                        },
                            React.createElement('th', { scope: "row" }, idx + 1),
                            React.createElement('td', null, JSON.stringify(row))
                        )
                    )
                )
            )
        )
    );
}
