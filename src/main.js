import { LogisticRegression, LinearRegression, accuracy, confusionMatrix, precisionScore, recallScore, f1Score, mse, rmse, mae, r2Score } from './models.js';
import { getDataset, getDatasetsByType } from './datasets.js';
import { setupCSVUpload } from './csv-upload.js';
import { tutorials } from './tutorials.js';

const state = {
  dataset: getDataset('synthetic'),
  model: null, Xtrain: [], ytrain: [], Xtest: [], ytest: [],
  means: null, stds: null, training: false, trained: false,
  modelType: 'classification',
  colors: ['#22d3ee', '#f472b6'],
  xMin: 0, xMax: 10, yMin: 0, yMax: 10
};

// DOM
const $ = id => document.getElementById(id);
const datasetSel = $('datasetSel');
const modelTypeSel = $('modelTypeSel');
const splitRange = $('splitRange');
const splitLabel = $('splitLabel');
const lrRange = $('lrRange');
const lrLabel = $('lrLabel');
const epochsInput = $('epochsInput');
const trainBtn = $('trainBtn');
const resetBtn = $('resetBtn');
const overlay = $('trainOverlay');
const epochBadge = $('epochBadge');
const costBadge = $('costBadge');
const sampleInfo = $('sampleInfo');
const featureInfo = $('featureInfo');
const classLabel0 = $('classLabel0');
const classLabel1 = $('classLabel1');
const metricsPanel = $('metricsPanel');
const classifMetrics = $('classifMetrics');
const regMetrics = $('regMetrics');
const confMatrixSection = $('confMatrixSection');
const predictPanel = $('predictPanel');
const mAcc = $('mAcc'), mPrec = $('mPrec'), mRec = $('mRec'), mF1 = $('mF1');
const mMSE = $('mMSE'), mRMSE = $('mRMSE'), mMAE = $('mMAE'), mR2 = $('mR2');
const cmTN = $('cmTN'), cmFP = $('cmFP'), cmFN = $('cmFN'), cmTP = $('cmTP');
const cfBias = $('cfBias'), cfW1 = $('cfW1'), cfW2 = $('cfW2'), cfFormula = $('cfFormula');
const predSlider1 = $('predSlider1'), predSlider2 = $('predSlider2');
const predVal1 = $('predVal1'), predVal2 = $('predVal2');
const predLabel1 = $('predLabel1'), predLabel2 = $('predLabel2');
const predOut = $('predOut');
const bCanvas = $('boundaryCanvas'), bCtx = bCanvas.getContext('2d');
const cCanvas = $('costCanvas'), cCtx = cCanvas.getContext('2d');
const tooltip = $('canvasTooltip');
const panelTitle = document.querySelector('.grid-2 .panel:first-child .panel-h h2');

// ── Model type ────────────────────────────────────

function rebuildDatasetOptions(type) {
  const datasets = getDatasetsByType(type);
  datasetSel.innerHTML = '';
  for (const d of datasets) {
    const opt = document.createElement('option');
    opt.value = d.value;
    opt.textContent = d.label;
    datasetSel.appendChild(opt);
  }
  datasetSel.value = datasets[0].value;
  applyModelTypeUI(type);
  loadDataset();
}

function applyModelTypeUI(type) {
  const isReg = type === 'regression';
  panelTitle.textContent = isReg ? 'Superficie de Regresión' : 'Frontera de Decisión';
  classLabel0.style.display = isReg ? 'none' : '';
  classLabel1.style.display = isReg ? 'none' : '';
  classifMetrics.style.display = isReg ? 'none' : '';
  regMetrics.style.display = isReg ? '' : 'none';
  confMatrixSection.style.display = isReg ? 'none' : '';
  const fn = state.dataset ? state.dataset.featureNames : ['x₁', 'x₂'];
  cfFormula.innerHTML = isReg
    ? `ŷ = β₀ + β₁·${fn[0]} + β₂·${fn[1]}`
    : `ŷ = σ(β₀ + β₁·${fn[0]} + β₂·${fn[1]})`;
  costBadge.textContent = isReg ? 'MSE: —' : 'Costo: —';
}

modelTypeSel.addEventListener('change', () => {
  state.modelType = modelTypeSel.value;
  state.model = null; state.trained = false;
  resetUI();
  rebuildDatasetOptions(state.modelType);
});

// ── Dataset ────────────────────────────────────

function loadDataset() {
  const ds = getDataset(datasetSel.value);
  if (!ds) return;
  state.dataset = ds;
  state.model = null; state.trained = false;
  const { samples, featureNames } = ds;
  sampleInfo.textContent = `Muestras: ${samples.length}`;
  featureInfo.textContent = `Features: ${featureNames.length}`;
  predLabel1.textContent = featureNames[0];
  predLabel2.textContent = featureNames[1];

  if (ds.classNames) {
    classLabel0.innerHTML = `<span class="dot c0"></span> ${ds.classNames[0]}`;
    classLabel1.innerHTML = `<span class="dot c1"></span> ${ds.classNames[1]}`;
  }

  let xM = Infinity, xX = -Infinity, yM = Infinity, yX = -Infinity;
  for (const s of samples) {
    xM = Math.min(xM, s.features[0]); xX = Math.max(xX, s.features[0]);
    yM = Math.min(yM, s.features[1]); yX = Math.max(yX, s.features[1]);
  }
  const xp = (xX - xM) * 0.15 || 0.5, yp = (yX - yM) * 0.15 || 0.5;
  state.xMin = xM - xp; state.xMax = xX + xp; state.yMin = yM - yp; state.yMax = yX + yp;

  predSlider1.min = state.xMin; predSlider1.max = state.xMax; predSlider1.value = (state.xMin + state.xMax) / 2;
  predSlider2.min = state.yMin; predSlider2.max = state.yMax; predSlider2.value = (state.yMin + state.yMax) / 2;
  predVal1.textContent = parseFloat(predSlider1.value).toFixed(1);
  predVal2.textContent = parseFloat(predSlider2.value).toFixed(1);

  resetUI();
  applyModelTypeUI(state.modelType);
  drawBoundary();
  drawCostChart([]);
}

function resetUI() {
  metricsPanel.style.display = 'none';
  predictPanel.style.display = 'none';
  resetBtn.disabled = true;
  trainBtn.disabled = false;
  epochBadge.textContent = '0 épocas';
  costBadge.textContent = state.modelType === 'regression' ? 'MSE: —' : 'Costo: —';
  state.trained = false;
}

// ── Training ───────────────────────────────────

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function doSplit() {
  const { samples } = state.dataset;
  const ratio = parseInt(splitRange.value) / 100;
  const shuffled = shuffle([...samples]);
  const idx = Math.floor(shuffled.length * ratio);
  state.Xtrain = shuffled.slice(0, idx).map(s => s.features);
  state.ytrain = shuffled.slice(0, idx).map(s => s.label);
  state.Xtest = shuffled.slice(idx).map(s => s.features);
  state.ytest = shuffled.slice(idx).map(s => s.label);
}

function computeStats(X) {
  const m = X[0].length, means = [], stds = [];
  for (let j = 0; j < m; j++) {
    let sum = 0;
    for (const row of X) sum += row[j];
    means.push(sum / X.length);
  }
  for (let j = 0; j < m; j++) {
    let ss = 0;
    for (const row of X) ss += Math.pow(row[j] - means[j], 2);
    stds.push(Math.sqrt(ss / X.length) || 1);
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
  overlay.classList.add('active');
  metricsPanel.style.display = 'none';
  predictPanel.style.display = 'none';

  doSplit();
  const stats = computeStats(state.Xtrain);
  state.means = stats.means; state.stds = stats.stds;
  const Xs = standardize(state.Xtrain, state.means, state.stds);
  const lr = parseInt(lrRange.value) / 100;
  const epochs = parseInt(epochsInput.value);
  const isReg = state.modelType === 'regression';
  const model = isReg ? new LinearRegression() : new LogisticRegression();
  const m = state.Xtrain[0].length;
  model.weights = new Array(m).fill(0);
  model.bias = 0;
  model.costHistory = [];

  await new Promise(resolve => {
    let completed = 0;
    const batchSize = 10;

    function step() {
      const end = Math.min(completed + batchSize, epochs);
      for (let e = completed; e < end; e++) {
        const preds = Xs.map(x => {
          let z = model.bias;
          for (let j = 0; j < m; j++) z += model.weights[j] * x[j];
          return isReg ? z : model.sigmoid(z);
        });

        let cost = 0;
        if (isReg) {
          for (let i = 0; i < Xs.length; i++) {
            const d = preds[i] - state.ytrain[i];
            cost += d * d;
          }
          cost /= Xs.length;
        } else {
          for (let i = 0; i < Xs.length; i++) {
            const eps = 1e-12;
            cost += -state.ytrain[i] * Math.log(preds[i] + eps) - (1 - state.ytrain[i]) * Math.log(1 - preds[i] + eps);
          }
          cost /= Xs.length;
        }
        model.costHistory.push(cost);

        const dw = new Array(m).fill(0);
        let db = 0;
        for (let i = 0; i < Xs.length; i++) {
          const err = preds[i] - state.ytrain[i];
          for (let j = 0; j < m; j++) dw[j] += err * Xs[i][j];
          db += err;
        }

        const scale = (isReg ? 2 * lr : lr) / Xs.length;
        for (let j = 0; j < m; j++) model.weights[j] -= scale * dw[j];
        model.bias -= scale * db;
      }

      completed = end;
      epochBadge.textContent = `${completed} épocas`;
      if (model.costHistory.length > 0) {
        const label = isReg ? 'MSE' : 'Costo';
        costBadge.textContent = `${label}: ${model.costHistory[model.costHistory.length - 1].toFixed(4)}`;
      }
      drawCostChart(model.costHistory);
      if (completed < epochs) setTimeout(step, 25);
      else { model.trained = true; state.model = model; resolve(); }
    }
    step();
  });

  overlay.classList.remove('active');
  state.training = false; state.trained = true;
  trainBtn.disabled = false; resetBtn.disabled = false;
  drawBoundary();
  showMetrics();
  metricsPanel.style.display = 'block';
  predictPanel.style.display = 'block';
}

function showMetrics() {
  const Xs = standardize(state.Xtest, state.means, state.stds);
  const preds = state.model.predict(Xs);
  const isReg = state.modelType === 'regression';

  if (isReg) {
    const m = mse(state.ytest, preds);
    mMSE.textContent = m.toFixed(4);
    mRMSE.textContent = rmse(state.ytest, preds).toFixed(4);
    mMAE.textContent = mae(state.ytest, preds).toFixed(4);
    mR2.textContent = r2Score(state.ytest, preds).toFixed(4);
  } else {
    const acc = accuracy(state.ytest, preds);
    const prec = precisionScore(state.ytest, preds);
    const rec = recallScore(state.ytest, preds);
    const f1 = f1Score(state.ytest, preds);
    const cm = confusionMatrix(state.ytest, preds);
    mAcc.textContent = acc.toFixed(3);
    mPrec.textContent = prec.toFixed(3);
    mRec.textContent = rec.toFixed(3);
    mF1.textContent = f1.toFixed(3);
    cmTN.textContent = cm.tn; cmFP.textContent = cm.fp;
    cmFN.textContent = cm.fn; cmTP.textContent = cm.tp;
  }

  cfBias.textContent = state.model.bias.toFixed(4);
  cfW1.textContent = state.model.weights[0].toFixed(4);
  cfW2.textContent = state.model.weights[1].toFixed(4);

  const fn = state.dataset.featureNames;
  const prefix = isReg ? '' : 'σ(';
  const suffix = isReg ? '' : ')';
  cfFormula.innerHTML = `ŷ = ${prefix}${state.model.bias.toFixed(3)} + ${state.model.weights[0].toFixed(3)}·${fn[0]} + ${state.model.weights[1].toFixed(3)}·${fn[1]}${suffix}`;

  doLivePredict();
}

// ── Canvas: boundary / surface ──────────────────

const BW = 500, BH = 400, BPAD = 40;

function drawBoundary() {
  bCtx.clearRect(0, 0, BW, BH);
  const { samples, featureNames } = state.dataset;
  if (!samples.length) return;

  const xMin = state.xMin, xMax = state.xMax, yMin = state.yMin, yMax = state.yMax;
  const sx = (BW - 2 * BPAD) / (xMax - xMin);
  const sy = (BH - 2 * BPAD) / (yMax - yMin);
  const toX = v => BPAD + (v - xMin) * sx;
  const toY = v => BH - BPAD - (v - yMin) * sy;
  state._sx = sx; state._sy = sy; state._toX = toX; state._toY = toY;

  const isReg = state.modelType === 'regression';

  if (isReg && state.trained) {
    const yVals = samples.map(s => s.label);
    const yMinR = Math.min(...yVals), yMaxR = Math.max(...yVals), yRange = yMaxR - yMinR || 1;
    for (let px = 0; px < BW; px += 3) {
      for (let py = 0; py < BH; py += 3) {
        const x = (px - BPAD) / sx + xMin;
        const y = (BH - BPAD - py) / sy + yMin;
        const xs = [(x - state.means[0]) / state.stds[0], (y - state.means[1]) / state.stds[1]];
        let z = state.model.bias;
        for (let j = 0; j < xs.length; j++) z += state.model.weights[j] * xs[j];
        const t = Math.max(0, Math.min(1, (z - yMinR) / yRange));
        const r = Math.round(255 * t);
        const b = Math.round(255 * (1 - t));
        bCtx.fillStyle = `rgba(${r},150,${b},0.12)`;
        bCtx.fillRect(px, py, 3, 3);
      }
    }
  } else if (!isReg && state.trained) {
    for (let px = 0; px < BW; px += 3) {
      for (let py = 0; py < BH; py += 3) {
        const x = (px - BPAD) / sx + xMin;
        const y = (BH - BPAD - py) / sy + yMin;
        const xs = [(x - state.means[0]) / state.stds[0], (y - state.means[1]) / state.stds[1]];
        let z = state.model.bias;
        for (let j = 0; j < xs.length; j++) z += state.model.weights[j] * xs[j];
        const prob = state.model.sigmoid(z);
        if (Math.abs(prob - 0.5) < 0.08) bCtx.fillStyle = 'rgba(255,255,255,0.1)';
        else bCtx.fillStyle = prob > 0.5
          ? `rgba(244,114,182,${Math.min((prob - 0.5) * 0.15, 0.08)})`
          : `rgba(34,211,238,${Math.min((0.5 - prob) * 0.15, 0.08)})`;
        bCtx.fillRect(px, py, 3, 3);
      }
    }
  }

  // Grid
  bCtx.strokeStyle = 'rgba(255,255,255,0.04)';
  bCtx.lineWidth = 1;
  for (let v = Math.ceil(xMin); v <= xMax; v++) {
    bCtx.beginPath(); bCtx.moveTo(toX(v), BPAD); bCtx.lineTo(toX(v), BH - BPAD); bCtx.stroke();
  }
  for (let v = Math.ceil(yMin); v <= yMax; v++) {
    bCtx.beginPath(); bCtx.moveTo(BPAD, toY(v)); bCtx.lineTo(BW - BPAD, toY(v)); bCtx.stroke();
  }
  bCtx.strokeStyle = 'rgba(255,255,255,0.1)';
  bCtx.lineWidth = 1;
  if (0 >= xMin && 0 <= xMax) { bCtx.beginPath(); bCtx.moveTo(toX(0), BPAD); bCtx.lineTo(toX(0), BH - BPAD); bCtx.stroke(); }
  if (0 >= yMin && 0 <= yMax) { bCtx.beginPath(); bCtx.moveTo(BPAD, toY(0)); bCtx.lineTo(BW - BPAD, toY(0)); bCtx.stroke(); }

  // Points
  if (isReg && state.trained) {
    const yVals = samples.map(s => s.label);
    const yMinR = Math.min(...yVals), yMaxR = Math.max(...yVals), yRange = yMaxR - yMinR || 1;
    for (const s of samples) {
      const cx = toX(s.features[0]), cy = toY(s.features[1]);
      const t = Math.max(0, Math.min(1, (s.label - yMinR) / yRange));
      const r = Math.round(255 * t);
      const b = Math.round(255 * (1 - t));
      const color = `rgb(${r},130,${b})`;
      bCtx.shadowColor = color; bCtx.shadowBlur = 7;
      bCtx.fillStyle = color;
      bCtx.beginPath(); bCtx.arc(cx, cy, 5, 0, Math.PI * 2); bCtx.fill();
      bCtx.shadowBlur = 0;
      bCtx.strokeStyle = 'rgba(255,255,255,0.25)'; bCtx.lineWidth = 1; bCtx.stroke();
    }
  } else {
    for (const s of samples) {
      const cx = toX(s.features[0]), cy = toY(s.features[1]);
      const color = state.colors[s.label];
      bCtx.shadowColor = color; bCtx.shadowBlur = 7;
      bCtx.fillStyle = color;
      bCtx.beginPath(); bCtx.arc(cx, cy, 5, 0, Math.PI * 2); bCtx.fill();
      bCtx.shadowBlur = 0;
      bCtx.strokeStyle = 'rgba(255,255,255,0.25)'; bCtx.lineWidth = 1; bCtx.stroke();
    }
  }

  // Labels
  bCtx.fillStyle = 'rgba(255,255,255,0.18)';
  bCtx.font = '9px Inter, sans-serif';
  bCtx.textAlign = 'center'; bCtx.textBaseline = 'top';
  for (let v = Math.ceil(xMin); v <= xMax; v++) { if (v === 0) continue; bCtx.fillText(v, toX(v), toY(0) + 6); }
  bCtx.textAlign = 'right'; bCtx.textBaseline = 'middle';
  for (let v = Math.ceil(yMin); v <= yMax; v++) { if (v === 0) continue; bCtx.fillText(v, toX(0) - 7, toY(v)); }
  bCtx.fillStyle = 'rgba(255,255,255,0.25)';
  bCtx.font = '10px Inter, sans-serif';
  bCtx.textAlign = 'left'; bCtx.textBaseline = 'bottom';
  bCtx.fillText(featureNames[0], BW - BPAD, toY(0) - 2);
  bCtx.textAlign = 'center'; bCtx.textBaseline = 'bottom';
  bCtx.fillText(featureNames[1], toX(0), BPAD - 2);
}

// ── Canvas: cost chart ─────────────────────────

const CW = 500, CH = 300, CPAD = 40;

function drawCostChart(history) {
  cCtx.clearRect(0, 0, CW, CH);
  if (history.length < 2) return;

  const minC = Math.min(...history), maxC = Math.max(...history), range = Math.max(maxC - minC, 0.01);
  const yBot = Math.max(0, minC - range * 0.1), yTop = maxC + range * 0.1;
  const maxE = history.length - 1;
  const sx = (CW - 2 * CPAD) / Math.max(maxE, 1);
  const sy = (CH - 2 * CPAD) / Math.max(yTop - yBot, 0.01);
  const toX = v => CPAD + v * sx;
  const toY = v => CH - CPAD - (v - yBot) * sy;

  // Grid
  cCtx.strokeStyle = 'rgba(255,255,255,0.04)';
  cCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = yBot + (yTop - yBot) * i / 4;
    cCtx.beginPath(); cCtx.moveTo(CPAD, toY(y)); cCtx.lineTo(CW - CPAD, toY(y)); cCtx.stroke();
  }

  // Fill
  cCtx.beginPath();
  cCtx.moveTo(toX(0), CH - CPAD);
  for (let i = 0; i < history.length; i++) cCtx.lineTo(toX(i), toY(history[i]));
  cCtx.lineTo(toX(maxE), CH - CPAD);
  cCtx.closePath();
  cCtx.fillStyle = 'rgba(124,58,237,0.07)';
  cCtx.fill();

  // Curve
  cCtx.strokeStyle = '#7c3aed';
  cCtx.lineWidth = 2;
  cCtx.shadowColor = '#7c3aed'; cCtx.shadowBlur = 8;
  cCtx.beginPath();
  for (let i = 0; i < history.length; i++) {
    i === 0 ? cCtx.moveTo(toX(i), toY(history[i])) : cCtx.lineTo(toX(i), toY(history[i]));
  }
  cCtx.stroke();
  cCtx.shadowBlur = 0;

  // Labels
  cCtx.fillStyle = 'rgba(255,255,255,0.2)';
  cCtx.font = '9px Inter, sans-serif';
  cCtx.textAlign = 'right'; cCtx.textBaseline = 'middle';
  cCtx.fillText(yTop.toFixed(2), CPAD - 5, toY(yTop));
  cCtx.fillText(yBot.toFixed(2), CPAD - 5, toY(yBot));
  cCtx.textAlign = 'center'; cCtx.textBaseline = 'top';
  cCtx.fillText(`${maxE} épocas`, toX(maxE), CH - CPAD + 5);
  cCtx.fillText('0', CPAD, CH - CPAD + 5);
}

// ── Prediction ─────────────────────────────────

function doLivePredict() {
  if (!state.trained) return;
  const f1 = parseFloat(predSlider1.value);
  const f2 = parseFloat(predSlider2.value);
  const xs = [(f1 - state.means[0]) / state.stds[0], (f2 - state.means[1]) / state.stds[1]];
  let z = state.model.bias;
  for (let j = 0; j < xs.length; j++) z += state.model.weights[j] * xs[j];

  if (state.modelType === 'regression') {
    predOut.innerHTML = `
      <div style="font-size:1.4rem;font-weight:800;font-family:var(--mono);color:var(--accent2)">${z.toFixed(4)}</div>
      <div style="font-size:0.78rem;color:var(--text3)">Valor predicho</div>
    `;
  } else {
    const prob = state.model.sigmoid(z);
    const cls = prob >= 0.5 ? 1 : 0;
    const name = state.dataset.classNames[cls];
    const color = state.colors[cls];
    const pct = (prob >= 0.5 ? prob : 1 - prob) * 100;
    predOut.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:4px">
        <div style="font-size:1.2rem;font-weight:700;color:${color}">${name}</div>
        <div style="font-size:0.85rem;color:var(--text2)">Confianza: <strong style="color:var(--accent2);font-family:var(--mono)">${pct.toFixed(1)}%</strong></div>
      </div>
    `;
  }
}

function predictAtPoint(x, y) {
  if (!state.trained) return;
  predSlider1.value = Math.min(state.xMax, Math.max(state.xMin, x));
  predSlider2.value = Math.min(state.yMax, Math.max(state.yMin, y));
  predVal1.textContent = parseFloat(predSlider1.value).toFixed(1);
  predVal2.textContent = parseFloat(predSlider2.value).toFixed(1);
  doLivePredict();
}

// ── Canvas interaction ─────────────────────────

bCanvas.addEventListener('mousemove', e => {
  const rect = bCanvas.getBoundingClientRect();
  const px = (e.clientX - rect.left) / rect.width * BW;
  const py = (e.clientY - rect.top) / rect.height * BH;
  const toX = state._toX, toY = state._toY;
  if (!toX) return;
  const x = (px - BPAD) / state._sx + state.xMin;
  const y = (BH - BPAD - py) / state._sy + state.yMin;
  if (x < state.xMin || x > state.xMax || y < state.yMin || y > state.yMax) {
    tooltip.classList.remove('show');
    return;
  }
  tooltip.style.left = (px + 12) + 'px';
  tooltip.style.top = (py - 10) + 'px';
  tooltip.textContent = `(${x.toFixed(2)}, ${y.toFixed(2)})`;
  tooltip.classList.add('show');
});

bCanvas.addEventListener('mouseleave', () => tooltip.classList.remove('show'));

bCanvas.addEventListener('click', e => {
  const rect = bCanvas.getBoundingClientRect();
  const px = (e.clientX - rect.left) / rect.width * BW;
  const py = (e.clientY - rect.top) / rect.height * BH;
  const x = (px - BPAD) / state._sx + state.xMin;
  const y = (BH - BPAD - py) / state._sy + state.yMin;
  predictAtPoint(x, y);
});

// ── Events ─────────────────────────────────────

datasetSel.addEventListener('change', loadDataset);
splitRange.addEventListener('input', () => { splitLabel.textContent = splitRange.value; });
lrRange.addEventListener('input', () => { lrLabel.textContent = (parseInt(lrRange.value) / 100).toFixed(2); });
trainBtn.addEventListener('click', trainModel);
resetBtn.addEventListener('click', () => {
  state.model = null; state.trained = false;
  resetUI();
  drawBoundary(); drawCostChart([]);
});
predSlider1.addEventListener('input', () => {
  predVal1.textContent = parseFloat(predSlider1.value).toFixed(1);
  doLivePredict();
});
predSlider2.addEventListener('input', () => {
  predVal2.textContent = parseFloat(predSlider2.value).toFixed(1);
  doLivePredict();
});

// ── Section Navigation ─────────────────────────

const navBtns = document.querySelectorAll('.nav-btn');
const sections = {
  home: document.getElementById('section-home'),
  app: document.getElementById('section-app'),
  tutorials: document.getElementById('section-tutorials')
};

function showSection(name) {
  Object.keys(sections).forEach(key => {
    sections[key].style.display = key === name ? '' : 'none';
  });
  navBtns.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.section === name);
  });
}

navBtns.forEach(btn => {
  btn.addEventListener('click', () => showSection(btn.dataset.section));
});

document.getElementById('heroCTA').addEventListener('click', () => showSection('app'));
document.getElementById('heroTutorials').addEventListener('click', () => showSection('tutorials'));

// ── Tutorials ──────────────────────────────────

function renderTutorials() {
  const grid = document.getElementById('tutorialsGrid');
  if (!grid) return;
  grid.innerHTML = tutorials.map(t => `
    <div class="tutorial-card" data-tutorial="${t.id}">
      <div class="tutorial-card-header">
        <span class="tutorial-card-title">${t.title}</span>
        <span class="tutorial-card-arrow">&#9654;</span>
      </div>
      <div class="tutorial-card-body">${t.content}</div>
    </div>
  `).join('');

  grid.querySelectorAll('.tutorial-card').forEach(card => {
    card.querySelector('.tutorial-card-header').addEventListener('click', () => {
      card.classList.toggle('expanded');
      const arrow = card.querySelector('.tutorial-card-arrow');
      arrow.innerHTML = card.classList.contains('expanded') ? '&#9660;' : '&#9654;';
    });
  });
}

renderTutorials();

// ── Init ───────────────────────────────────────

function init() {
  rebuildDatasetOptions(state.modelType);

  setupCSVUpload((name) => {
    let exists = false;
    for (const opt of datasetSel.options) {
      if (opt.value === name) { exists = true; break; }
    }
    if (!exists) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      datasetSel.appendChild(opt);
    }
    datasetSel.value = name;
    loadDataset();
  });
}

init();
