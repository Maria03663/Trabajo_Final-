import { animate } from 'motion';

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const slopeInput = document.getElementById('slope');
const interceptInput = document.getElementById('intercept');
const colorInput = document.getElementById('lineColor');

const formulaDisplay = document.getElementById('formulaDisplay');
const mDisplay = document.getElementById('mDisplay');
const bDisplay = document.getElementById('bDisplay');
const coordDisplay = document.getElementById('coordDisplay');
const colorDot = document.getElementById('colorDot');

const W = canvas.width;
const H = canvas.height;
const pad = 40;

const xMin = -10, xMax = 10, yMin = -10, yMax = 10;
const scaleX = (W - 2 * pad) / (xMax - xMin);
const scaleY = (H - 2 * pad) / (yMax - yMin);

function toCanvasX(x) { return pad + (x - xMin) * scaleX; }
function toCanvasY(y) { return H - pad - (y - yMin) * scaleY; }

function draw() {
  ctx.clearRect(0, 0, W, H);

  const color = colorInput.value;
  const m = parseFloat(slopeInput.value) || 0;
  const b = parseFloat(interceptInput.value) || 0;

  // Grid
  ctx.strokeStyle = 'rgba(100, 140, 200, 0.08)';
  ctx.lineWidth = 1;
  for (let x = Math.ceil(xMin); x <= xMax; x++) {
    if (x === 0) continue;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(x), pad);
    ctx.lineTo(toCanvasX(x), H - pad);
    ctx.stroke();
  }
  for (let y = Math.ceil(yMin); y <= yMax; y++) {
    if (y === 0) continue;
    ctx.beginPath();
    ctx.moveTo(pad, toCanvasY(y));
    ctx.lineTo(W - pad, toCanvasY(y));
    ctx.stroke();
  }

  // Axes
  ctx.shadowColor = 'rgba(100, 180, 255, 0.3)';
  ctx.shadowBlur = 6;
  ctx.strokeStyle = 'rgba(100, 180, 255, 0.5)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(pad, toCanvasY(0));
  ctx.lineTo(W - pad, toCanvasY(0));
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(toCanvasX(0), pad);
  ctx.lineTo(toCanvasX(0), H - pad);
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Axis arrows
  ctx.fillStyle = 'rgba(100, 180, 255, 0.5)';
  ctx.beginPath();
  ctx.moveTo(W - pad, toCanvasY(0));
  ctx.lineTo(W - pad - 8, toCanvasY(0) - 4);
  ctx.lineTo(W - pad - 8, toCanvasY(0) + 4);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(toCanvasX(0), pad);
  ctx.lineTo(toCanvasX(0) - 4, pad + 8);
  ctx.lineTo(toCanvasX(0) + 4, pad + 8);
  ctx.fill();

  // Axis labels
  ctx.fillStyle = 'rgba(150, 200, 255, 0.7)';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let x = Math.ceil(xMin); x <= xMax; x++) {
    if (x === 0) continue;
    ctx.fillText(x, toCanvasX(x), toCanvasY(0) + 8);
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let y = Math.ceil(yMin); y <= yMax; y++) {
    if (y === 0) continue;
    ctx.fillText(y, toCanvasX(0) - 10, toCanvasY(y));
  }

  // Axis labels
  ctx.fillStyle = 'rgba(100, 180, 255, 0.6)';
  ctx.font = '12px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillText('x', toCanvasX(9.5), toCanvasY(0) - 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText('y', toCanvasX(0), pad + 2);

  // Origin
  ctx.fillStyle = 'rgba(100, 180, 255, 0.5)';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.font = '11px Inter, sans-serif';
  ctx.fillText('O', toCanvasX(0) - 8, toCanvasY(0) + 8);

  // Line with glow
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.beginPath();
  const x1 = xMin, y1 = m * x1 + b;
  const x2 = xMax, y2 = m * x2 + b;
  if (isFinite(y1) && isFinite(y2)) {
    ctx.moveTo(toCanvasX(x1), toCanvasY(y1));
    ctx.lineTo(toCanvasX(x2), toCanvasY(y2));
    ctx.stroke();
  }
  ctx.shadowBlur = 0;

  // Highlight intercept
  const bCanvasY = toCanvasY(b);
  if (b >= yMin && b <= yMax) {
    ctx.shadowColor = color;
    ctx.shadowBlur = 15;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(toCanvasX(0), bCanvasY, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }

  // Update info
  const mStr = m.toFixed(2);
  const bStr = b.toFixed(2);
  const sign = b >= 0 ? '+' : '-';
  formulaDisplay.innerHTML = `y = <span style="color:#22d3ee">${mStr}</span>x ${sign} <span style="color:#a78bfa">${Math.abs(bStr)}</span>`;
  mDisplay.textContent = mStr;
  bDisplay.textContent = bStr;
  coordDisplay.textContent = `(0, ${bStr})`;
  colorDot.style.backgroundColor = color;
}

// Mouse tracking
canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  const canvasPx = (px / rect.width) * W;
  const canvasPy = (py / rect.height) * H;
  const x = (canvasPx - pad) / scaleX + xMin;
  const y = yMax - (canvasPy - pad) / scaleY;
  if (x >= xMin && x <= xMax && y >= yMin && y <= yMax) {
    canvas.title = `(${x.toFixed(2)}, ${y.toFixed(2)})`;
  }
});

[slopeInput, interceptInput, colorInput].forEach(el =>
  el.addEventListener('input', draw)
);

// Motion entrance animations
const enterConfig = { duration: 0.5, easing: [0.25, 0.1, 0.25, 1] };

animate(
  document.getElementById('header'),
  { opacity: [0, 1], y: [20, 0] },
  { ...enterConfig, delay: 0 }
);

animate(
  document.getElementById('controls'),
  { opacity: [0, 1], y: [16, 0] },
  { ...enterConfig, delay: 0.15 }
);

animate(
  document.getElementById('canvasWrapper'),
  { opacity: [0, 1], y: [24, 0] },
  { ...enterConfig, delay: 0.3 }
);

animate(
  document.getElementById('info'),
  { opacity: [0, 1], y: [16, 0] },
  { ...enterConfig, delay: 0.45 }
);

draw();
