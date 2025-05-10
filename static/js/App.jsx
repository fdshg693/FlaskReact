const { useState, useEffect } = React;

function App() {
  const [msg, setMsg] = useState('読み込み中…');
  const [count, setCount] = useState(0);

  useEffect(() => {
    fetchMessage()
      .then(setMsg)
      .catch(() => setMsg('取得失敗'));
  }, []);

  // 複雑ロジック例：5秒ごとにカウント+1
  useEffect(() => {
    const id = setInterval(() => setCount(c => c + 1), 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{ fontFamily: 'sans-serif', padding: '20px' }}>
      <Message text={msg} />
      <p>カウント: {count}</p>
      <Button onClick={() => alert('クリック！')}>押してみて</Button>
      <Image></Image>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
