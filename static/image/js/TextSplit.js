import React from 'react';
import { splitTextAPI } from './api.js';

const { useState, useEffect } = React;

export function TextSplit({ rawText, setRawText }) {
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

    return React.createElement('div', { className: "max-w-md mx-auto p-6 bg-blue-300 rounded-2xl shadow-md space-y-4" },
        React.createElement('h3', { className: "text-2xl font-bold mb-4" }, 'テキスト分割'),
        React.createElement('p', { className: "text-2xl font-bold mb-4" }, '分割したいテキストを入力してください'),
        React.createElement('textarea', {
            rows: "10",
            className: "w-full p-2 border border-gray-300 rounded mb-4",
            placeholder: "ここにテキストを入力してください",
            value: rawText,
            onChange: (e) => setRawText(e.target.value)
        }),
        React.createElement('button', { onClick: () => setClicked(true) }, '分割'),
        splitText && React.createElement(React.Fragment, null,
            React.createElement('div', { className: "split-text text-lg text-gray-700 mt-4" },
                '分割されたテキスト：',
                React.createElement('br'),
                splitText.map((chunk, index) =>
                    React.createElement('div', {
                        key: index,
                        className: "mb-2 p-2 bg-white rounded shadow"
                    }, chunk)
                )
            )
        )
    );
}
