export async function trainUserData(inputs) {
    const res = await fetch('/api/userData', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputs),
    });
    if (!res.ok) throw new Error('APIエラー');
    const data = await res.json();
    if (data.userData == "finished") {
        alert("学習が完了しました。");
    } else if (data.userData == "error") {
        alert("学習に失敗しました。");
    }
}