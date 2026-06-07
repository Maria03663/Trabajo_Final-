import { registerCustomDataset } from './datasets.js';

let parsedData = null;
let fileInput = null;

export function setupCSVUpload(onLoaded) {
  fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = '.csv';
  fileInput.style.display = 'none';
  document.body.appendChild(fileInput);

  fileInput.addEventListener('change', handleFileSelect);
  document.getElementById('uploadBtn').addEventListener('click', () => fileInput.click());
  document.getElementById('modalCancel').addEventListener('click', closeModal);
  document.getElementById('modalConfirm').addEventListener('click', () => confirmUpload(onLoaded));

  document.getElementById('modalLabelCol').addEventListener('change', validateModal);
  document.getElementById('modalFeat1Col').addEventListener('change', validateModal);
  document.getElementById('modalFeat2Col').addEventListener('change', validateModal);
}

async function handleFileSelect() {
  const file = fileInput.files[0];
  if (!file) return;
  const text = await file.text();
  parsedData = parseCSV(text);
  if (!parsedData || parsedData.rows.length < 10) {
    showToast('El CSV debe tener al menos 10 filas de datos.');
    fileInput.value = '';
    return;
  }
  showModal(file.name);
}

function showModal(fileName) {
  const fname = fileName.replace(/\.[^.]+$/, '');
  document.getElementById('modalFileName').textContent = fileName;
  document.getElementById('modalDatasetName').value = fname;

  const labelSel = document.getElementById('modalLabelCol');
  const feat1Sel = document.getElementById('modalFeat1Col');
  const feat2Sel = document.getElementById('modalFeat2Col');
  labelSel.innerHTML = '';
  feat1Sel.innerHTML = '';
  feat2Sel.innerHTML = '';

  for (const h of parsedData.headers) {
    labelSel.appendChild(opt(h, h));
    feat1Sel.appendChild(opt(h, h));
    feat2Sel.appendChild(opt(h, h));
  }

  const last = parsedData.headers.length - 1;
  labelSel.value = parsedData.headers[last];
  feat1Sel.value = parsedData.headers[0];
  feat2Sel.value = parsedData.headers[Math.min(1, last)];

  document.getElementById('modalError').textContent = '';
  document.getElementById('modalClassInfo').innerHTML = '';
  document.getElementById('modalOverlay').style.display = 'flex';
  document.getElementById('modalConfirm').disabled = true;
  validateModal();
}

function validateModal() {
  const labelCol = document.getElementById('modalLabelCol').value;
  const feat1Col = document.getElementById('modalFeat1Col').value;
  const feat2Col = document.getElementById('modalFeat2Col').value;
  const errEl = document.getElementById('modalError');
  const infoEl = document.getElementById('modalClassInfo');
  const confirmBtn = document.getElementById('modalConfirm');

  if (feat1Col === labelCol || feat2Col === labelCol || feat1Col === feat2Col) {
    errEl.textContent = 'Las columnas deben ser distintas entre sí.';
    confirmBtn.disabled = true;
    infoEl.innerHTML = '';
    return;
  }

  for (const col of [feat1Col, feat2Col]) {
    for (const row of parsedData.rows.slice(0, 30)) {
      if (row[col] !== '' && isNaN(Number(row[col]))) {
        errEl.textContent = `La columna "${col}" contiene valores no numéricos.`;
        confirmBtn.disabled = true;
        infoEl.innerHTML = '';
        return;
      }
    }
  }

  errEl.textContent = '';
  confirmBtn.disabled = false;

  const uniqueLabels = new Set(parsedData.rows.map(r => r[labelCol]));
  if (uniqueLabels.size === 2) {
    const sorted = [...uniqueLabels].sort();
    infoEl.innerHTML = `
      <span style="font-size:0.72rem;color:var(--text3);margin-bottom:2px">Tipo: <strong style="color:var(--accent2)">Clasificación</strong></span>
      <span class="modal-info-item"><span class="dot c0"></span> Clase 0: <strong style="color:var(--text)">${escHtml(sorted[0])}</strong></span>
      <span class="modal-info-item"><span class="dot c1"></span> Clase 1: <strong style="color:var(--text)">${escHtml(sorted[1])}</strong></span>
    `;
  } else {
    infoEl.innerHTML = `
      <span style="font-size:0.72rem;color:var(--text3);margin-bottom:2px">Tipo: <strong style="color:var(--accent2)">Regresión</strong></span>
      <span class="modal-info-item">Etiqueta con <strong style="color:var(--text)">${uniqueLabels.size}</strong> valores distintos</span>
    `;
  }
}

function confirmUpload(onLoaded) {
  const name = document.getElementById('modalDatasetName').value.trim();
  if (!name) {
    document.getElementById('modalError').textContent = 'Ingresá un nombre para el dataset.';
    return;
  }

  const labelCol = document.getElementById('modalLabelCol').value;
  const feat1Col = document.getElementById('modalFeat1Col').value;
  const feat2Col = document.getElementById('modalFeat2Col').value;

  const uniqueLabels = [...new Set(parsedData.rows.map(r => r[labelCol]))];
  const isClassif = uniqueLabels.length === 2;
  const type = isClassif ? 'classification' : 'regression';

  let labelMap = {};
  if (isClassif) {
    const sorted = [...uniqueLabels].sort();
    labelMap = { [sorted[0]]: 0, [sorted[1]]: 1 };
  }

  const samples = [];
  for (const row of parsedData.rows) {
    const f1 = Number(row[feat1Col]);
    const f2 = Number(row[feat2Col]);
    if (isNaN(f1) || isNaN(f2)) continue;
    const lbl = isClassif ? labelMap[row[labelCol]] : Number(row[labelCol]);
    if (lbl === undefined || isNaN(lbl)) continue;
    samples.push({ features: [f1, f2], label: lbl });
  }

  if (samples.length < 10) {
    document.getElementById('modalError').textContent = 'No hay suficientes filas válidas.';
    return;
  }

  const dataset = {
    samples,
    featureNames: [feat1Col, feat2Col],
    type
  };
  if (isClassif) dataset.classNames = [...uniqueLabels].sort();

  registerCustomDataset(name, dataset, type);
  closeModal();
  onLoaded(name);
}

function closeModal() {
  document.getElementById('modalOverlay').style.display = 'none';
  fileInput.value = '';
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return null;

  const delimiter = detectDelimiter(lines);
  const headers = parseLine(lines[0], delimiter);
  const rows = [];

  for (let i = 1; i < lines.length; i++) {
    const vals = parseLine(lines[i], delimiter);
    if (vals.length !== headers.length) continue;
    const row = {};
    let valid = true;
    for (let j = 0; j < headers.length; j++) {
      const v = vals[j].trim();
      if (v === '') { valid = false; break; }
      row[headers[j]] = v;
    }
    if (valid) rows.push(row);
  }
  return { headers, rows };
}

function parseLine(line, delimiter) {
  const result = [];
  let cur = '';
  let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') { inQ = !inQ; continue; }
    if (ch === delimiter && !inQ) { result.push(cur); cur = ''; }
    else cur += ch;
  }
  result.push(cur);
  return result;
}

function detectDelimiter(lines) {
  let comma = 0, semi = 0, tab = 0;
  for (const line of lines.slice(0, 5)) {
    comma += (line.match(/,/g) || []).length;
    semi += (line.match(/;/g) || []).length;
    tab += (line.match(/\t/g) || []).length;
  }
  if (tab >= comma && tab >= semi) return '\t';
  if (semi > comma) return ';';
  return ',';
}

function opt(value, text) {
  const o = document.createElement('option');
  o.value = value;
  o.textContent = text;
  return o;
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function showToast(msg) {
  const el = document.createElement('div');
  el.style.cssText = 'position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#f87171;color:#fff;padding:10px 20px;border-radius:8px;font-size:0.82rem;z-index:200;font-family:var(--font)';
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}
