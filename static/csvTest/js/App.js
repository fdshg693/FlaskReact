import React from 'react';
import { createRoot } from 'react-dom/client';
import { CsvLoader } from './CsvLoader.js';

const { useState, useEffect } = React;

function App() {
  return React.createElement('div', { style: { fontFamily: 'sans-serif', padding: '20px' } },
    React.createElement(CsvLoader)
  );
}

const root = createRoot(document.getElementById('root'));
root.render(React.createElement(App));
