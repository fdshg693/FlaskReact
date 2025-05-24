//画像判定APIを呼び出す
async function analyzeImage(selectedFile) {
    const formData = new FormData();
    formData.append('image', selectedFile, selectedFile.name);

    const res = await fetch('/api/image', {
        method: 'POST',
        headers: {
            // 'Content-Type' は FormData を使うと自動で設定されるため不要
        },
        body: formData,
    });
    if (!res.ok) throw new Error('APIエラー');
    const data = await res.json();
    return data.description;
}

// PDF認識APIを呼び出す
async function analyzePDF(selectedFile) {
    const formData = new FormData();
    formData.append('pdf', selectedFile, selectedFile.name);

    const res = await fetch('/api/pdf', {
        method: 'POST',
        headers: {
            // 'Content-Type' は FormData を使うと自動で設定されるため不要
        },
        body: formData,
    });
    if (!res.ok) throw new Error('APIエラー');
    const data = await res.json();
    return data.text[0].page_content;
}