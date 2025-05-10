async function fetchImage() {
    const res = await fetch('/api/image');
    if (!res.ok) throw new Error('APIエラー');
    const data = await res.json();
    return data.text;
  }
  