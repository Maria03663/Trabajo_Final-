import { animate } from 'motion';
import { LogisticRegression, accuracy, confusionMatrix, precisionScore, recallScore, f1Score } from './logistic-regression.js';
import { getDataset } from './datasets.js';

let state = {
  dataset: getDataset('synthetic'),
  model: null,
  Xtrain: [],
  ytrain: [],
  Xtest: [],
  ytest: [],
  means: null,
  stds: null,
  training: false,
  colors: ['#22d3ee', '#f472b6']
};

const boundaryCanvas = document.getElementById('boundaryCanvas');
const bCtx = boundaryCanvas.getContext('2d');
const costCanvas = document.getElementById('costCanvas');
const cCtx = costCanvas.getContext('2d');

// Motion animations
animate('#header', { opacity: [0, 1], y: [20, 0] }, { duration: 0.5, easing: [0.25, 0.1, 0.25, 1], delay: 0 });
animate('#controlBar', { opacity: [0, 1], y: [16, 0] }, { duration: 0.5, easing: [0.25, 0.1, 0.25, 1], delay: 0.12 });
animate('#sampleInfo', { opacity: [0, 1], y: [12, 0] }, { duration: 0.5, easing: [0.25, 0.1, 0.25, 1], delay: 0.2 });
animate('.main-grid', { opacity: [0, 1], y: [24, 0] }, { duration: 0.5, easing: [0.25, 0.1, 0.25, 1], delay: 0.3 });

// DOM refs
const datasetSelect = document.getElementById('datasetSelect');
const trainSplit = document.getElementById('trainSplit');
const trainSplitLabel = document.getElementById('trainSplitLabel');
const learningRate = document.getElementById('learningRate');
const lrLabel = document.getElementById('lrLabel');
const epochsInput = document.getElementById('epochs');
const trainBtn = document.getElementById('trainBtn');
const resetBtn = document.getElementById('resetBtn');
const predictBtn = document.getElementById('predictBtn');
const trainingOverlay = document.getElementById('trainingOverlay');
const epochBadge = document.getElementById('epochBadge');
const costBadge = document.getElementById('costBadge');
const panelMetrics = document.getElementById('panelMetrics');
const panelPredict = document.getElementById('panelPredict');
const sampleCount = document.getElementById('sampleCount');
const featureCount = document.getElementById('featureCount');

// Metric displays
const metricAccuracy = document.getElementById('metricAccuracy');
const metricPrecision = document.getElementById('metricPrecision');
const metricRecall = document.getElementById('metricRecall');
const metricF1 = document.getElementById('metricF1');
const cmTN = document.getElementById('cmTN');
const cmFP = document.getElementById('cmFP');
const cmFN = document.getElementById('cmFN');
const cmTP = document.getElementById('cmTP');
const coefBias = document.getElementById('coefBias');
const coefW1 = document.getElementById('coefW1');
const coefW2 = document.getElementById('coefW2');
const coeffFormula = document.getElementById('coeffFormula');
const predictResult = document.getElementById('predictResult');
const predF1 = document.getElementById('predF1');
const predF2 = document.getElementById('predF2');

function updateDataset() {
  const name = datasetSelect.value;
  state.dataset = getDataset(name);
  state.model = null;
  state.training = false;

  const { samples, featureNames } = state.dataset;
  sampleCount.textContent = `Muestras: ${samples.length}`;
  featureCount.textContent = `Features: ${featureNames.length}`;

  predF1.placeholder = featureNames[0] || 'x₁';
  predF2.placeholder = featureNames[1] || 'x₂';

  panelMetrics.style.display = 'none';
  panelPredict.style.display = 'none';
  resetBtn.disabled = true;
  trainBtn.disabled = false;
  epochBadge.textContent = '0 épocas';
  costBadge.textContent = 'Costo: —';

  drawBoundary();
}

function splitData() {
  const { samples } = state.dataset;
  const ratio = parseInt(trainSplit.value) / 100;
  const shuffled = [...samples];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const splitIdx = Math.floor(shuffled.length * ratio);
  const train = shuffled.slice(0, splitIdx);
  const test = shuffled.slice(splitIdx);

  state.Xtrain = train.map(s => s.features);
  state.ytrain = train.map(s => s.label);
  state.Xtest = test.map(s => s.features);
  state.ytest = test.map(s => s.label);
}

function computeStats(X) {
  const m = X[0].length;
  const means = [], stds = [];
  for (let j = 0; j < m; j++) {
    let sum = 0;
    for (const row of X) sum += row[j];
    means.push(sum / X.length);
  }
  for (let j = 0; j < m; j++) {
    let sumSq = 0;
    for (const row of X) sumSq += Math.pow(row[j] - means[j], 2);
    stds.push(Math.sqrt(sumSq / X.length) || 1);
  }
  return { means, stds };
}

function standardize(X, means, stds) {
  return X.map(row => row.map((v, j) => (v - means[j]) / (stds[j] || 1)));
}

async function trainModel() {
  if (state.training) return;
  state.training = true;
  trainBtn.disabled = true;
  trainingOverlay.classList.add('active');
  panelMetrics.style.display = 'none';
  panelPredict.style.display = 'none';

  splitData();
  const stats = computeStats(state.Xtrain);
  state.means = stats.means;
  state.stds = stats.stds;
  const Xs = standardize(state.Xtrain, state.means, state.stds);

  const lr = parseInt(learningRate.value) / 100;
  const epochs = parseInt(epochsInput.value);
  const model = new LogisticRegression();

  await new Promise(resolve => {
    const batchSize = 10;
    let completed = 0;

    function step() {
      const end = Math.min(completed + batchSize, epochs);
      for (let e = completed; e < end; e++) {
        const preds = Xs.map(x => {
          let z = model.bias;
          for (let j = 0; j < model.weights.length; j++) z += model.weights[j] * x[j];
          return model.sigmoid(z);
        });
        let cost = 0;
        for (let i = 0; i < Xs.length; i++) {
          const eps = 1e-12;
          cost += -state.ytrain[i] * Math.log(preds[i] + eps) - (1 - state.ytrain[i]) * Math.log(1 - preds[i] + eps);
        }
        model.costHistory.push(cost / Xs.length);

        const dw = new Array(model.weights.length).fill(0);
        let db = 0;
        for (let i = 0; i < Xs.length; i++) {
          const err = preds[i] - state.ytrain[i];
          for (let j = 0; j < model.weights.length; j++) dw[j] += err * Xs[i][j];
          db += err;
        }
        for (let j = 0; j < model.weights.length; j++) model.weights[j] -= (lr / Xs.length) * dw[j];
        model.bias -= (lr / Xs.length) * db;
      }
      completed = end;

      epochBadge.textContent = `${completed} épocas`;
      if (model.costHistory.length > 0) {
        costBadge.textContent = `Costo: ${model.costHistory[model.costHistory.length - 1].toFixed(4)}`;
      }
      drawCostChart(model.costHistory);

      if (completed < epochs) {
        setTimeout(step, 30);
      } else {
        model.trained = true;
        state.model = model;
        state.weights = model.weights;
        state.bias = model.bias;
        resolve();
      }
    }
    step();
  });

  trainingOverlay.classList.remove('active');
  state.training = false;
  trainBtn.disabled = false;
  resetBtn.disabled = false;

  drawBoundary();
  showMetrics();
  panelMetrics.style.display = 'block';
  panelPredict.style.display = 'block';

  animate(panelMetrics, { opacity: [0, 1], y: [16, 0] }, { duration: 0.4, easing: [0.25, 0.1, 0.25, 1] });
  animate(panelPredict, { opacity: [0, 1], y: [16, 0] }, { duration: 0.4, easing: [0.25, 0.1, 0.25, 1] });
}

function showMetrics() {
  const Xs = standardize(state.Xtest, state.means, state.stds);
  const preds = state.model.predict(Xs);

  const acc = accuracy(state.ytest, preds);
  const prec = precisionScore(state.ytest, preds);
  const rec = recallScore(state.ytest, preds);
  const f1 = f1Score(state.ytest, preds);
  const cm = confusionMatrix(state.ytest, preds);

  metricAccuracy.textContent = acc.toFixed(3);
  metricPrecision.textContent = prec.toFixed(3);
  metricRecall.textContent = rec.toFixed(3);
  metricF1.textContent = f1.toFixed(3);

  cmTN.textContent = cm.tn;
  cmFP.textContent = cm.fp;
  cmFN.textContent = cm.fn;
  cmTP.textContent = cm.tp;

  coefBias.textContent = state.model.bias.toFixed(4);
  coefW1.textContent = state.model.weights[0].toFixed(4);
  coefW2.textContent = state.model.weights[1].toFixed(4);

  const fn = state.dataset.featureNames;
  coeffFormula.innerHTML = `ŷ = σ(${state.model.bias.toFixed(3)} + ${state.model.weights[0].toFixed(3)}·${fn[0]} + ${state.model.weights[1].toFixed(3)}·${fn[1]})`;
}

function drawBoundary() {
  const W = 500, H = 400;
  const pad = 40;
  bCtx.clearRect(0, 0, W, H);

  const { samples, featureNames } = state.dataset;
  if (samples.length === 0) return;

  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const s of samples) {
    xMin = Math.min(xMin, s.features[0]);
    xMax = Math.max(xMax, s.features[0]);
    yMin = Math.min(yMin, s.features[1]);
    yMax = Math.max(yMax, s.features[1]);
  }
  const xPad = (xMax - xMin) * 0.15 || 0.5;
  const yPad = (yMax - yMin) * 0.15 || 0.5;
  xMin -= xPad; xMax += xPad; yMin -= yPad; yMax += yPad;

  const sx = (W - 2 * pad) / (xMax - xMin);
  const sy = (H - 2 * pad) / (yMax - yMin);

  function toX(v) { return pad + (v - xMin) * sx; }
  function toY(v) { return H - pad - (v - yMin) * sy; }

  if (state.model && state.model.trained) {
    const res = 2;
    for (let px = 0; px < W; px += res) {
      for (let py = 0; py < H; py += res) {
        const x = (px - pad) / sx + xMin;
        const y = (H - pad - py) / sy + yMin;
        const xs = [(x - state.means[0]) / state.stds[0], (y - state.means[1]) / state.stds[1]];
        let z = state.model.bias;
        for (let j = 0; j < xs.length; j++) z += state.model.weights[j] * xs[j];
        const prob = state.model.sigmoid(z);
        if (Math.abs(prob - 0.5) < 0.08) {
          bCtx.fillStyle = `rgba(255,255,255,0.15)`;
          bCtx.fillRect(px, py, res, res);
        } else {
          bCtx.fillStyle = prob > 0.5
            ? `rgba(244, 114, 182, ${Math.min((prob - 0.5) * 0.15, 0.08)})`
            : `rgba(34, 211, 238, ${Math.min((0.5 - prob) * 0.15, 0.08)})`;
          bCtx.fillRect(px, py, res, res);
        }
      }
    }
  }

  // Grid
  bCtx.strokeStyle = 'rgba(255,255,255,0.04)';
  bCtx.lineWidth = 1;
  for (let v = Math.ceil(xMin); v <= xMax; v++) {
    bCtx.beginPath(); bCtx.moveTo(toX(v), pad); bCtx.lineTo(toX(v), H - pad); bCtx.stroke();
  }
  for (let v = Math.ceil(yMin); v <= yMax; v++) {
    bCtx.beginPath(); bCtx.moveTo(pad, toY(v)); bCtx.lineTo(W - pad, toY(v)); bCtx.stroke();
  }

  // Axes
  bCtx.strokeStyle = 'rgba(255,255,255,0.12)';
  bCtx.lineWidth = 1;
  if (0 >= xMin && 0 <= xMax) {
    bCtx.beginPath(); bCtx.moveTo(toX(0), pad); bCtx.lineTo(toX(0), H - pad); bCtx.stroke();
  }
  if (0 >= yMin && 0 <= yMax) {
    bCtx.beginPath(); bCtx.moveTo(pad, toY(0)); bCtx.lineTo(W - pad, toY(0)); bCtx.stroke();
  }

  // Data points
  for (const s of samples) {
    const cx = toX(s.features[0]);
    const cy = toY(s.features[1]);
    const color = state.colors[s.label];
    const r = s.label === 0 ? 5 : 5;

    bCtx.shadowColor = color;
    bCtx.shadowBlur = 6;
    bCtx.fillStyle = color;
    bCtx.beginPath();
    bCtx.arc(cx, cy, r, 0, Math.PI * 2);
    bCtx.fill();
    bCtx.shadowBlur = 0;

    bCtx.strokeStyle = 'rgba(255,255,255,0.3)';
    bCtx.lineWidth = 1;
    bCtx.stroke();
  }

  // Axis labels
  bCtx.fillStyle = 'rgba(255,255,255,0.25)';
  bCtx.font = '10px Inter, sans-serif';
  bCtx.textAlign = 'center';
  bCtx.textBaseline = 'top';
  for (let v = Math.ceil(xMin); v <= xMax; v++) {
    if (v === 0) continue;
    bCtx.fillText(v, toX(v), toY(0) + 6);
  }
  bCtx.textAlign = 'right';
  bCtx.textBaseline = 'middle';
  for (let v = Math.ceil(yMin); v <= yMax; v++) {
    if (v === 0) continue;
    bCtx.fillText(v, toX(0) - 8, toY(v));
  }

  bCtx.fillStyle = 'rgba(255,255,255,0.3)';
  bCtx.font = '10px Inter, sans-serif';
  bCtx.textAlign = 'left';
  bCtx.textBaseline = 'bottom';
  bCtx.fillText(featureNames[0], W - pad, toY(0) - 2);
  bCtx.textAlign = 'center';
  bCtx.textBaseline = 'bottom';
  bCtx.fillText(featureNames[1], toX(0), pad - 2);
}

function drawCostChart(costHistory) {
  const W = 500, H = 300;
  const pad = 40;
  cCtx.clearRect(0, 0, W, H);

  if (costHistory.length < 2) return;

  const minCost = Math.min(...costHistory);
  const maxCost = Math.max(...costHistory);
  const cRange = Math.max(maxCost - minCost, 0.01);
  const yMinC = Math.max(0, minCost - cRange * 0.1);
  const yMaxC = maxCost + cRange * 0.1;
  const xMaxE = costHistory.length - 1;

  const sx = (W - 2 * pad) / Math.max(xMaxE, 1);
  const sy = (H - 2 * pad) / Math.max(yMaxC - yMinC, 0.01);

  function toCx(v) { return pad + v * sx; }
  function toCy(v) { return H - pad - (v - yMinC) * sy; }

  // Grid
  cCtx.strokeStyle = 'rgba(255,255,255,0.04)';
  cCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = yMinC + (yMaxC - yMinC) * i / 4;
    cCtx.beginPath(); cCtx.moveTo(pad, toCy(y)); cCtx.lineTo(W - pad, toCy(y)); cCtx.stroke();
  }

  // Fill under curve
  cCtx.beginPath();
  cCtx.moveTo(toCx(0), toCy(0));
  for (let i = 0; i < costHistory.length; i++) {
    cCtx.lineTo(toCx(i), toCy(costHistory[i]));
  }
  cCtx.lineTo(toCx(xMaxE), toCy(0));
  cCtx.closePath();
  cCtx.fillStyle = 'rgba(124, 58, 237, 0.1)';
  cCtx.fill();

  // Cost curve
  cCtx.strokeStyle = '#7c3aed';
  cCtx.lineWidth = 2;
  cCtx.shadowColor = '#7c3aed';
  cCtx.shadowBlur = 8;
  cCtx.beginPath();
  for (let i = 0; i < costHistory.length; i++) {
    if (i === 0) cCtx.moveTo(toCx(i), toCy(costHistory[i]));
    else cCtx.lineTo(toCx(i), toCy(costHistory[i]));
  }
  cCtx.stroke();
  cCtx.shadowBlur = 0;

  // Labels
  cCtx.fillStyle = 'rgba(255,255,255,0.25)';
  cCtx.font = '10px Inter, sans-serif';
  cCtx.textAlign = 'right';
  cCtx.textBaseline = 'middle';
  cCtx.fillText(yMaxC.toFixed(2), pad - 6, toCy(yMaxC));
  cCtx.fillText(yMinC.toFixed(2), pad - 6, toCy(yMinC));
  cCtx.textAlign = 'center';
  cCtx.textBaseline = 'top';
  cCtx.fillText(`${xMaxE} épocas`, toCx(xMaxE), H - pad + 6);
  cCtx.fillText('0', pad, H - pad + 6);
}

// Prediction
function predict() {
  if (!state.model || !state.model.trained) return;

  const f1 = parseFloat(predF1.value);
  const f2 = parseFloat(predF2.value);
  if (isNaN(f1) || isNaN(f2)) {
    predictResult.innerHTML = '<div class="result-placeholder">Ingresá ambos valores</div>';
    return;
  }

  const xs = [(f1 - state.means[0]) / state.stds[0], (f2 - state.means[1]) / state.stds[1]];
  let z = state.model.bias;
  for (let j = 0; j < xs.length; j++) z += state.model.weights[j] * xs[j];
  const prob = state.model.sigmoid(z);
  const predClass = prob >= 0.5 ? 1 : 0;
  const className = state.dataset.classNames[predClass];
  const color = state.colors[predClass];

  predictResult.innerHTML = `
    <div class="result-content">
      <div class="result-class" style="color:${color}">${className}</div>
      <div class="result-prob">Probabilidad: <span>${(prob >= 0.5 ? prob : 1 - prob).toFixed(1)}%</span></div>
    </div>
  `;
}

// Event listeners
datasetSelect.addEventListener('change', updateDataset);
trainSplit.addEventListener('input', () => {
  trainSplitLabel.textContent = trainSplit.value;
});
learningRate.addEventListener('input', () => {
  lrLabel.textContent = (parseInt(learningRate.value) / 100).toFixed(2);
});
trainBtn.addEventListener('click', trainModel);
resetBtn.addEventListener('click', () => {
  state.model = null;
  panelMetrics.style.display = 'none';
  panelPredict.style.display = 'none';
  resetBtn.disabled = true;
  epochBadge.textContent = '0 épocas';
  costBadge.textContent = 'Costo: —';
  drawBoundary();
  cCtx.clearRect(0, 0, costCanvas.width, costCanvas.height);
});
predictBtn.addEventListener('click', predict);

// Init
updateDataset();
drawCostChart([]);
