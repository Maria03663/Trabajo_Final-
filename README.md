# Classifica

Aplicación educativa interactiva para visualizar y entender **regresión logística** y **regresión lineal** desde cero.

Entrená modelos con descenso de gradiente paso a paso, observá la frontera de decisión / superficie de regresión en vivo y evaluá métricas — todo desde el navegador, sin backend.

## Funcionalidades

- **Dos modelos:** Regresión Logística (clasificación binaria) y Regresión Lineal (predicción continua)
- **Datasets predefinidos:** Sintético 2D, Iris (Setosa vs Versicolor), Sintético Regresión — filtrados por tipo de modelo
- **Carga de CSV:** subí tu propio dataset, seleccioná columnas y auto‑detecta si es clasificación o regresión
- **Entrenamiento visual:** animación del descenso de gradiente época por época
- **Frontera de decisión / Superficie de regresión:** mapa de calor de probabilidades o plano de regresión sobre los datos
- **Gráfico de costo:** evolución de la pérdida (entropía cruzada o MSE) durante el entrenamiento
- **Métricas de clasificación:** accuracy, precision, recall, F1‑score y matriz de confusión
- **Métricas de regresión:** MSE, RMSE, MAE y R²
- **Predicción en vivo:** sliders interactivos o click en el scatter plot
- **Tutoriales integrados:** 6 guías educativas expandibles sobre los conceptos clave
- **Landing page:** presentación del proyecto con hero, tarjetas de features y estadísticas

## Tecnologías

- [Vite](https://vitejs.dev/) — bundler y dev server
- JavaScript vanilla (ES modules, sin frameworks)
- Canvas API para visualizaciones

## Estructura del proyecto

```
src/
├── main.js            → entrada, lógica de entrenamiento, navegación SPA
├── style.css          → diseño oscuro completo
├── models.js          → LogisticRegression, LinearRegression y métricas
├── datasets.js        → registro de datasets, filtrado por tipo
├── csv-upload.js      → parseo y validación de CSV
└── tutorials.js       → contenido educativo de los tutoriales
```

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
