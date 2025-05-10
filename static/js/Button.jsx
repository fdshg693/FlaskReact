function Button({ onClick, children }) {
    return (
      <button onClick={onClick} style={{ padding: '8px 12px', cursor: 'pointer' }}>
        {children}
      </button>
    );
  }
  