const { useState, useEffect } = React;

function App() {
    const [splitTextRaw, setSplitTextRaw] = useState('');
    return (
        <div className="font-sans p-5">
            <div className="flex space-x-5">
                <div className="w-1/2">
                    <Image />
                </div>
                <div className="w-1/2">
                    <PDF />
                </div>
                <div className="w-1/2">
                    <TextSplit rawText={splitTextRaw} setRawText={setSplitTextRaw} />
                </div>
            </div>
        </div>
    );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
