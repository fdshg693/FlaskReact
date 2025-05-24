//画像判定APIを呼び出す
async function fetchImage() {
    const res = await fetch('/api/image');
    if (!res.ok) throw new Error('APIエラー');
    const data = await res.json();
    return data.letter;
}