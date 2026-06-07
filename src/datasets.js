const datasetEntries = {
  synthetic: { fn: syntheticDataset, type: 'classification', label: 'Sintético 2D' },
  iris: { fn: irisDataset, type: 'classification', label: 'Iris (Setosa vs Versicolor)' },
  'synthetic-reg': { fn: syntheticRegressionDataset, type: 'regression', label: 'Sintético Regresión' },
};

const customDatasets = new Map();

export function getDataset(name) {
  if (customDatasets.has(name)) {
    return structuredClone(customDatasets.get(name).dataset);
  }
  const entry = datasetEntries[name];
  if (!entry) return syntheticDataset();
  return entry.fn();
}

export function getDatasetsByType(type) {
  const result = [];
  for (const [value, entry] of Object.entries(datasetEntries)) {
    if (entry.type === type) {
      result.push({ value, label: entry.label });
    }
  }
  for (const [name, data] of customDatasets) {
    if (data.type === type) {
      result.push({ value: name, label: name });
    }
  }
  return result;
}

export function registerCustomDataset(name, dataset, type) {
  customDatasets.set(name, { dataset, type });
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
    classNames: ['Clase 0', 'Clase 1'],
    type: 'classification'
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
    classNames: ['Setosa', 'Versicolor'],
    type: 'classification'
  };
}

function syntheticRegressionDataset() {
  const data = [];
  const r = rng(42);
  for (let i = 0; i < 50; i++) {
    const x1 = r() * 10;
    const x2 = r() * 10;
    const y = 3 + 2.5 * x1 - 1.8 * x2 + (r() - 0.5) * 3;
    data.push({ features: [x1, x2], label: y });
  }
  return {
    samples: data,
    featureNames: ['Feature 1', 'Feature 2'],
    type: 'regression'
  };
}
