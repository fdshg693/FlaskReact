async function fetchImage() {
  const res = await fetch('/api/image');
  if (!res.ok) throw new Error('APIエラー');
  const data = await res.json();
  return data.letter;
}

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