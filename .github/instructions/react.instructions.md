---
applyTo: '*.jsx'
---
## Current Environment
- React is used via CDN (Content Delivery Network) in the HTML file

## React import
- in HTML file, React file is imported like this
```html
<script type="text/babel" src="home/js/App.jsx"></script>
```
- in JSX file, React is imported like this
```javascript
const { useState, useEffect } = React;
```
- in JSX file, ReactDOM is used like this
```javascript
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
```

## React Development Guidelines
- Use functional components with hooks (useState, useEffect, etc.) for state management and side effects.
- Use JSX syntax for rendering UI components.
- Use `useState` for managing local component state.
- Use `useEffect` for side effects like data fetching or subscriptions.
- Use `props` to pass data from parent components to child components.
- Use `useContext` for global state management if needed.