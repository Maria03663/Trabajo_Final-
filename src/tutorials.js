export const tutorials = [
  {
    id: 'logistic',
    title: '¿Qué es la regresión logística?',
    content: `La <strong>regresión logística</strong> es un algoritmo de clasificación binaria. A diferencia de la regresión lineal, el resultado se pasa por una función <strong>sigmoide</strong> que lo comprime entre 0 y 1, interpretándose como una probabilidad.<br><br>
    <code>σ(z) = 1 / (1 + e<sup>−z</sup>)</code><br><br>
    Si la probabilidad es ≥ 0.5 se predice la clase 1; si es menor, la clase 0. El modelo aprende los coeficientes β (pesos) que minimizan la pérdida de <strong>entropía cruzada binaria</strong>.<br><br>
    Se usa en diagnóstico médico, detección de spam, aprobación de créditos y cualquier problema donde la salida sea "sí o no".`
  },
  {
    id: 'linear',
    title: '¿Qué es la regresión lineal?',
    content: `La <strong>regresión lineal</strong> predice un valor numérico continuo a partir de una combinación lineal de las características de entrada. El modelo ajusta un hiperplano (una recta en 2D, un plano en 3D) que minimiza el error cuadrático medio.<br><br>
    <code>ŷ = β₀ + β₁·x₁ + β₂·x₂ + ... + β<sub>n</sub>·x<sub>n</sub></code><br><br>
    La función de costo es el <strong>MSE</strong> (Mean Squared Error), y el descenso de gradiente actualiza los coeficientes para minimizarlo.<br><br>
    Se usa para predecir precios, ventas, temperaturas y cualquier variable continua.`
  },
  {
    id: 'gradient',
    title: 'Descenso de gradiente',
    content: `El <strong>descenso de gradiente</strong> es el algoritmo de optimización que ajusta los coeficientes del modelo. En cada época (iteración) se calcula el gradiente de la función de costo y se actualizan los parámetros en la dirección opuesta.<br><br>
    <code>βⱼ := βⱼ − α · (∂J/∂βⱼ)</code><br><br>
    Donde <strong>α</strong> (learning rate) controla el tamaño del paso. Si es muy grande, el modelo diverge; si es muy chico, converge lentamente.<br><br>
    En esta app podés ver el costo disminuir época por época en el gráfico de entrenamiento.`
  },
  {
    id: 'classif-metrics',
    title: 'Métricas de clasificación',
    content: `Para evaluar un clasificador binario se comparan las predicciones contra los valores reales usando la <strong>matriz de confusión</strong>:<br><br>
    <ul>
      <li><strong>Accuracy</strong> — aciertos totales / total de muestras</li>
      <li><strong>Precision</strong> — verdaderos positivos / (verdaderos positivos + falsos positivos)</li>
      <li><strong>Recall</strong> — verdaderos positivos / (verdaderos positivos + falsos negativos)</li>
      <li><strong>F1-Score</strong> — media armónica de precision y recall</li>
    </ul><br>
    Accuracy es útil cuando las clases están balanceadas. F1-Score es mejor cuando hay desbalanceo.`
  },
  {
    id: 'reg-metrics',
    title: 'Métricas de regresión',
    content: `Para evaluar un modelo de regresión se miden los errores entre los valores reales y los predichos:<br><br>
    <ul>
      <li><strong>MSE</strong> (Error Cuadrático Medio) — penaliza errores grandes al elevarlos al cuadrado</li>
      <li><strong>RMSE</strong> — raíz cuadrada del MSE, en las mismas unidades que la variable objetivo</li>
      <li><strong>MAE</strong> (Error Absoluto Medio) — promedio del valor absoluto de los errores</li>
      <li><strong>R²</strong> (Coeficiente de Determinación) — proporción de la varianza explicada por el modelo (1 = ajuste perfecto, 0 = modelo constante)</li>
    </ul><br>
    RMSE y MAE miden el error promedio; R² mide qué tan bien se ajusta el modelo a los datos.`
  },
  {
    id: 'howto',
    title: 'Cómo usar Classifica',
    content: `<strong>1. Elegí un modelo</strong><br>
    Seleccioná <em>Regresión Logística</em> para clasificación binaria o <em>Regresión Lineal</em> para predicción continua.<br><br>
    <strong>2. Seleccioná un dataset</strong><br>
    Usá los datasets precargados o subí tu propio CSV con el botón "Subir CSV". Elegí qué columnas usar como features y cuál como etiqueta.<br><br>
    <strong>3. Ajustá los hiperparámetros</strong><br>
    Modificá el porcentaje de train/test, la tasa de aprendizaje (α) y la cantidad de épocas.<br><br>
    <strong>4. Entrená</strong><br>
    Hacé click en "Entrenar" y observá el descenso de gradiente en vivo. El gráfico de costo muestra la convergencia.<br><br>
    <strong>5. Evaluá y predecí</strong><br>
    Revisá las métricas y la matriz de confusión. Usá los sliders o hacé click en el gráfico para predecir nuevos valores.`
  }
];
