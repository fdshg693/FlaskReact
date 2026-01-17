---
applyTo: '**/*.js, **/*.jsx'
---
## Current Environment
- React is used via ES Modules with Import Maps in the HTML file

## React import
- in HTML file, Import Maps are configured like this:
```html
<script type="importmap">
{
  "imports": {
    "react": "https://esm.sh/react@18",
    "react-dom/client": "https://esm.sh/react-dom@18/client",
    "papaparse": "https://esm.sh/papaparse@5.3.2"
  }
}
</script>
<script type="module" src="home/js/App.js"></script>
```
- in JS file, React is imported like this:
```javascript
import React from 'react';
import { createRoot } from 'react-dom/client';
const { useState, useEffect } = React;
```
- Components are exported and imported using ES modules:
```javascript
export function ComponentName() { ... }
import { ComponentName } from './ComponentName.js';
```
- JSX is replaced with React.createElement calls:
```javascript
return React.createElement('div', { className: "example" }, 'Hello World');
```

## React Development Guidelines
- Use functional components with hooks (useState, useEffect, etc.) for state management and side effects.
- Use React.createElement() instead of JSX syntax for rendering UI components.
- Use `useState` for managing local component state.
- Use `useEffect` for side effects like data fetching or subscriptions.
- Use `props` to pass data from parent components to child components.
- Use `useContext` for global state management if needed.
- Export components as named exports and import them explicitly.
- Use ES Modules import/export syntax for better tree shaking and modern JavaScript practices.