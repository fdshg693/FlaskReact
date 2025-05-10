// fetch系の共通処理をまとめる
async function fetchMessage() {
    const res = await fetch('/api/message');
    if (!res.ok) throw new Error('APIエラー');
    const data = await res.json();
    return data.text;
  }
  