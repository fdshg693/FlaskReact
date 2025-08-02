const { useState, useEffect } = React;

function Iris() {
  const speciesList = ['setosa', 'versicolor', 'virginica'];
  // State for each measurement input
  const [sepalLength, setSepalLength] = useState('');
  const [sepalWidth, setSepalWidth] = useState('');
  const [petalLength, setPetalLength] = useState('');
  const [petalWidth, setPetalWidth] = useState('');

  // State for predicted species
  const [species, setSpecies] = useState('');

  // ボタンがクリックされたかどうかを管理する state
  const [clicked, setClicked] = useState(false);

  // Handler to generate a random species
  useEffect(() => {
    if (!clicked) return; // ボタンがクリックされていない場合は何もしない
    // Validate inputs: ensure one decimal place
    const inputs = [sepalLength, sepalWidth, petalLength, petalWidth];
    const isValid = inputs.every(val => /^\d+(\.\d)?$/.test(val));
    if (!isValid) {
      alert('すべての値を小数点以下1桁の数値で入力してください。例: 5.1');
      setClicked(false);
      return;
    }
    (async () => {
      try {
        // fetchImage は fileUrl からサーバーに投げるなど適宜実装
        const result = await fetchIrisSpecies(inputs);
        console.log(result);
        setSpecies(result);
      } catch (e) {
        setSpecies('取得失敗');
      }
    })();
    setClicked(false);
  }, [clicked]);

  return (
    <div className="max-w-md mx-auto p-6 bg-white rounded-2xl shadow-md space-y-4">
      <h2 className="text-2xl font-semibold text-center">Iris Species Predictor</h2>

      <div className="space-y-2">
        <label className="block">
          <span>Sepal Length (cm):</span>
          <input
            type="number"
            step="0.1"
            value={sepalLength}
            onChange={e => setSepalLength(e.target.value)}
            placeholder="例: 5.1"
            className="mt-1 block w-full border rounded p-2"
          />
        </label>
        <label className="block">
          <span>Sepal Width (cm):</span>
          <input
            type="number"
            step="0.1"
            value={sepalWidth}
            onChange={e => setSepalWidth(e.target.value)}
            placeholder="例: 3.5"
            className="mt-1 block w-full border rounded p-2"
          />
        </label>
        <label className="block">
          <span>Petal Length (cm):</span>
          <input
            type="number"
            step="0.1"
            value={petalLength}
            onChange={e => setPetalLength(e.target.value)}
            placeholder="例: 1.4"
            className="mt-1 block w-full border rounded p-2"
          />
        </label>
        <label className="block">
          <span>Petal Width (cm):</span>
          <input
            type="number"
            step="0.1"
            value={petalWidth}
            onChange={e => setPetalWidth(e.target.value)}
            placeholder="例: 0.2"
            className="mt-1 block w-full border rounded p-2"
          />
        </label>
      </div>

      <button
        onClick={() => setClicked(true)}
        className="w-full py-2 rounded-2xl text-white font-medium bg-blue-500 hover:bg-blue-600 shadow"
      >
        Predict Species
      </button>

      {!species && (
        <p className="text-center text-xl font-semibold text-gray-500">
          ボタンを押してください
        </p>
      )}
      {species && (
        <p className="text-center text-xl font-semibold">
          Predicted species: <span className="text-green-600">{species}</span>
        </p>
      )}
    </div>
  );
}
