import React from 'react';
import { analyzePDF } from './api.js';
import { TextSplit } from './TextSplit.js';

const { useState, useEffect } = React;

export function PDF() {
    const [fileUrl, setFileUrl] = useState(null);
    const [description, setDescription] = useState('');
    const [selectedFile, setSelectedFile] = useState(null);
    const [splitTextRaw, setSplitTextRaw] = useState('');

    // fileUrl が変わったらPDF認識を呼び出す
    useEffect(() => {
        if (!fileUrl) return;

        (async () => {
            try {
                const result = await analyzePDF(selectedFile);
                setDescription(result);
            } catch (e) {
                setDescription('取得失敗');
            }
        })();
    }, [fileUrl]);

    // descriptionが変わったらsplitTextRawも更新
    useEffect(() => {
        setSplitTextRaw(description);
    }, [description]);

    // ファイル選択時のハンドラ
    const handleChange = (e) => {
        const file = e.target.files[0];
        setSelectedFile(file);
        // プレビュー用 URL を state に保存
        setFileUrl(URL.createObjectURL(file));
        // description は次の useEffect でセットされる
        setDescription('…認識中…');
    };

    return React.createElement('div', { className: "max-w-md mx-auto p-6 bg-blue-300 rounded-2xl shadow-md space-y-4" },
        React.createElement('h3', { className: "text-2xl font-bold mb-4" }, 'PDF文字起こし'),
        React.createElement('p', { className: "text-2xl font-bold mb-4" }, '文字起こししたいPDFをアップロードしてください'),
        React.createElement('input', {
            type: "file",
            accept: "application/pdf",
            onChange: handleChange,
            className: "mb-4"
        }),
        fileUrl && React.createElement(React.Fragment, null,
            React.createElement('div', { className: "description text-lg text-gray-700" },
                'PDFの内容：', description
            ),
            React.createElement(TextSplit, { rawText: splitTextRaw, setRawText: setSplitTextRaw })
        )
    );
}
