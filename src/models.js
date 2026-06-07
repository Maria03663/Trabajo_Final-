export class LogisticRegression {
  constructor() {
    this.weights = null;
    this.bias = null;
    this.costHistory = [];
    this.trained = false;
  }

  sigmoid(z) {
    return 1 / (1 + Math.exp(-Math.max(-100, Math.min(100, z))));
  }

  fit(X, y, { learningRate = 0.1, epochs = 1000 } = {}) {
    const n = X.length;
    const m = X[0].length;
    this.weights = new Array(m).fill(0);
    this.bias = 0;
    this.costHistory = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      const preds = new Array(n);
      for (let i = 0; i < n; i++) {
        let z = this.bias;
        for (let j = 0; j < m; j++) z += this.weights[j] * X[i][j];
        preds[i] = this.sigmoid(z);
      }
      let cost = 0;
      for (let i = 0; i < n; i++) {
        const eps = 1e-12;
        cost += -y[i] * Math.log(preds[i] + eps) - (1 - y[i]) * Math.log(1 - preds[i] + eps);
      }
      this.costHistory.push(cost / n);
      const dw = new Array(m).fill(0);
      let db = 0;
      for (let i = 0; i < n; i++) {
        const err = preds[i] - y[i];
        for (let j = 0; j < m; j++) dw[j] += err * X[i][j];
        db += err;
      }
      for (let j = 0; j < m; j++) this.weights[j] -= (learningRate / n) * dw[j];
      this.bias -= (learningRate / n) * db;
    }
    this.trained = true;
    return this;
  }

  predictProba(X) {
    if (!this.trained) throw new Error('Modelo no entrenado');
    return X.map(x => {
      let z = this.bias;
      for (let j = 0; j < this.weights.length; j++) z += this.weights[j] * x[j];
      return this.sigmoid(z);
    });
  }

  predict(X, threshold = 0.5) {
    return this.predictProba(X).map(p => p >= threshold ? 1 : 0);
  }
}

export function accuracy(yTrue, yPred) {
  let c = 0;
  for (let i = 0; i < yTrue.length; i++) if (yTrue[i] === yPred[i]) c++;
  return c / yTrue.length;
}

export function confusionMatrix(yTrue, yPred) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === 1 && yPred[i] === 1) tp++;
    else if (yTrue[i] === 0 && yPred[i] === 0) tn++;
    else if (yTrue[i] === 0 && yPred[i] === 1) fp++;
    else fn++;
  }
  return { tp, tn, fp, fn };
}

export function precisionScore(yTrue, yPred) {
  const { tp, fp } = confusionMatrix(yTrue, yPred);
  return tp + fp === 0 ? 0 : tp / (tp + fp);
}

export function recallScore(yTrue, yPred) {
  const { tp, fn } = confusionMatrix(yTrue, yPred);
  return tp + fn === 0 ? 0 : tp / (tp + fn);
}

export function f1Score(yTrue, yPred) {
  const p = precisionScore(yTrue, yPred);
  const r = recallScore(yTrue, yPred);
  return p + r === 0 ? 0 : 2 * p * r / (p + r);
}

export class LinearRegression {
  constructor() {
    this.weights = null;
    this.bias = null;
    this.costHistory = [];
    this.trained = false;
  }

  fit(X, y, { learningRate = 0.01, epochs = 1000 } = {}) {
    const n = X.length;
    const m = X[0].length;
    this.weights = new Array(m).fill(0);
    this.bias = 0;
    this.costHistory = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      const preds = new Array(n);
      for (let i = 0; i < n; i++) {
        let z = this.bias;
        for (let j = 0; j < m; j++) z += this.weights[j] * X[i][j];
        preds[i] = z;
      }
      let cost = 0;
      for (let i = 0; i < n; i++) {
        const d = preds[i] - y[i];
        cost += d * d;
      }
      this.costHistory.push(cost / n);
      const dw = new Array(m).fill(0);
      let db = 0;
      for (let i = 0; i < n; i++) {
        const err = preds[i] - y[i];
        for (let j = 0; j < m; j++) dw[j] += err * X[i][j];
        db += err;
      }
      const scale = 2 * learningRate / n;
      for (let j = 0; j < m; j++) this.weights[j] -= scale * dw[j];
      this.bias -= scale * db;
    }
    this.trained = true;
    return this;
  }

  predict(X) {
    if (!this.trained) throw new Error('Modelo no entrenado');
    return X.map(x => {
      let z = this.bias;
      for (let j = 0; j < this.weights.length; j++) z += this.weights[j] * x[j];
      return z;
    });
  }
}

export function mse(yTrue, yPred) {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const d = yTrue[i] - yPred[i];
    s += d * d;
  }
  return s / yTrue.length;
}

export function rmse(yTrue, yPred) {
  return Math.sqrt(mse(yTrue, yPred));
}

export function mae(yTrue, yPred) {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) s += Math.abs(yTrue[i] - yPred[i]);
  return s / yTrue.length;
}

export function r2Score(yTrue, yPred) {
  const mean = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < yTrue.length; i++) {
    ssRes += (yTrue[i] - yPred[i]) ** 2;
    ssTot += (yTrue[i] - mean) ** 2;
  }
  return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
}
