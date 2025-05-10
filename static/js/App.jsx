const { useState, useEffect } = React;

function App() {
  const [msg, setMsg] = useState('読み込み中…');
  const [count, setCount] = useState(0);

  return (
    <div style={{ fontFamily: 'sans-serif', padding: '20px' }}>
      <Image></Image>
      <Iris></Iris>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
