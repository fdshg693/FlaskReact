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

//CSVから学習するAPIを呼び出す
async function fetchMultipleIrisSpecies(inputs) {
  const res = await fetch('/api/batch_iris', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(inputs),
  });
  if (!res.ok) throw new Error('APIエラー');
  const data = await res.json();
  return data.userData
}