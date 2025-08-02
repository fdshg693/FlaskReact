import React from 'react';
import { createRoot } from 'react-dom/client';
import { Iris } from './Iris.js';
import { MultipleIris } from './MultipleIris.js';

const { useState, useEffect } = React;

function App() {
  return React.createElement('div', { className: "font-sans p-5" },
    React.createElement(Iris),
    React.createElement(MultipleIris)
  );
}

const root = createRoot(document.getElementById('root'));
root.render(React.createElement(App));
