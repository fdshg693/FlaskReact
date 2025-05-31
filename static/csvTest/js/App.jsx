const { useState, useEffect } = React;

function App() {
  return (
    <div style={{ fontFamily: 'sans-serif', padding: '20px' }}>
      <CsvLoader></CsvLoader>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
