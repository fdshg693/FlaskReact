//画像判定APIを呼び出す
async function fetchImage() {
  const res = await fetch('/api/image');
  if (!res.ok) throw new Error('APIエラー');
  const data = await res.json();
  return data.letter;
}

//アイリスのデータを判定するAPIを呼び出す
async function fetchIrisSpecies(inputs) {
  const res = await fetch('/api/iris', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(inputs),
  });
  if (!res.ok) throw new Error('APIエラー');
  const data = await res.json();
  return data.species;
}

//ユーザデータを学習するAPIを呼び出す
async function fetchMultipleIrisSpecies(inputs) {
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