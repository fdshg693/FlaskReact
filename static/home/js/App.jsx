const { useState, useEffect } = React;

function App() {
  return (
    <div className="font-sans p-5">
      <Iris />
      <MultipleIris />
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
