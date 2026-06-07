export function getDataset(name) {
  switch (name) {
    case 'synthetic': return syntheticDataset();
    case 'iris': return irisDataset();
    default: return syntheticDataset();
  }
}

function rng(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) | 0; return (s >>> 0) / 4294967296; };
}

function syntheticDataset() {
  const data = [];
  const r = rng(42);
  for (let i = 0; i < 25; i++) data.push({ features: [2 + r() * 1.8, 2 + r() * 1.8], label: 0 });
  for (let i = 0; i < 25; i++) data.push({ features: [6 + r() * 1.8, 6 + r() * 1.8], label: 1 });
  return {
    samples: data,
    featureNames: ['Feature 1', 'Feature 2'],
    classNames: ['Clase 0', 'Clase 1']
  };
}

function irisDataset() {
  const setosa = [
    [1.4,0.2],[1.3,0.2],[1.5,0.3],[1.6,0.4],[1.4,0.2],
    [1.7,0.3],[1.5,0.4],[1.3,0.3],[1.6,0.2],[1.4,0.3],
    [1.5,0.2],[1.7,0.4],[1.3,0.2],[1.6,0.3],[1.4,0.2],
    [1.5,0.3],[1.6,0.4],[1.4,0.2],[1.7,0.3],[1.5,0.4]
  ];
  const versicolor = [
    [4.2,1.3],[4.5,1.5],[4.9,1.7],[4.0,1.2],[4.6,1.4],
    [4.4,1.5],[4.8,1.8],[4.1,1.3],[4.3,1.4],[4.7,1.6],
    [4.5,1.4],[4.9,1.6],[4.2,1.2],[4.6,1.5],[4.3,1.3],
    [4.8,1.7],[4.1,1.4],[4.4,1.3],[4.7,1.5],[4.5,1.6]
  ];
  const data = [];
  for (const f of setosa) data.push({ features: f.slice(), label: 0 });
  for (const f of versicolor) data.push({ features: f.slice(), label: 1 });
  return {
    samples: data,
    featureNames: ['Largo pétalo', 'Ancho pétalo'],
    classNames: ['Setosa', 'Versicolor']
  };
}
