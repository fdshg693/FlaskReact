import React from 'react';
import { createRoot } from 'react-dom/client';
import { Image } from './Image.js';
import { PDF } from './PDF.js';
import { TextSplit } from './TextSplit.js';

const { useState, useEffect } = React;

function App() {
    const [splitTextRaw, setSplitTextRaw] = useState('');
    return React.createElement('div', { className: "font-sans p-5" },
        React.createElement('div', { className: "flex space-x-5" },
            React.createElement('div', { className: "w-1/2" },
                React.createElement(Image)
            ),
            React.createElement('div', { className: "w-1/2" },
                React.createElement(PDF)
            ),
            React.createElement('div', { className: "w-1/2" },
                React.createElement(TextSplit, { rawText: splitTextRaw, setRawText: setSplitTextRaw })
            )
        )
    );
}

const root = createRoot(document.getElementById('root'));
root.render(React.createElement(App));
