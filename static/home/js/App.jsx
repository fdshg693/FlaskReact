const { useState, useEffect } = React;

function App() {
  return (
    <div style={{ fontFamily: 'sans-serif', padding: '20px' }}>
      <Iris></Iris>
      <MultipleIris></MultipleIris>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
