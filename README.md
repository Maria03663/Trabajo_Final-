# Classifica

Aplicación educativa interactiva para visualizar y entender la **regresión logística** desde cero.

Entrena un clasificador binario con descenso de gradiente paso a paso, observá la frontera de decisión en vivo y evaluá las métricas — todo desde el navegador, sin backend.

## Funcionalidades

- **Datasets predefinidos:** Sintético 2D (clusters) e Iris (Setosa vs Versicolor)
- **Entrenamiento visual:** animación del descenso de gradiente época por época
- **Frontera de decisión:** mapa de calor de probabilidades superpuesto en los datos
- **Gráfico de costo:** evolución de la pérdida durante el entrenamiento
- **Métricas:** accuracy, precision, recall, F1-score y matriz de confusión
- **Predicción en vivo:** sliders que muestran clase predicha y confianza
- **Interacción:** hacé clic en el scatter plot para predecir cualquier punto

## Tecnologías

- [Vite](https://vitejs.dev/) — bundler y dev server
- JavaScript vanilla (ES modules, sin frameworks)
- Canvas API para visualizaciones
- [Motion](https://motion.dev/) — animaciones

## Cómo empezar

```bash
npm install
npm run dev
```

Abrí `http://localhost:5173` en el navegador.

## Build para producción

```bash
npm run build
npm run preview
```

## Deploy

El proyecto incluye configuración para DigitalOcean App Platform (`.do/app.yaml`). El build genera los archivos estáticos en `dist/`.
