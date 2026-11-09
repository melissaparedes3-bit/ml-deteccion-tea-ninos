
# Detección temprana de rasgos del espectro autista en niños pequeños

Proyecto final de Machine Learning: *Del documento técnico al prototipo inteligente*

Maestría en IA Aplicada a la Educación e Investigación (UFHEC)

**Autora:** Francis Melissa Paredes Hernández

## Descripción

Modelo de clasificación que, a partir de datos de tamizaje, estima la probabilidad de rasgos del espectro autista (TEA) en niños pequeños. Es una herramienta de apoyo y **no sustituye un diagnóstico profesional**.

## Dataset

- Fuente: Autism dataset (Kaggle)
- Archivo incluido: `archive.zip`

## Contenido del repositorio

- `archive.zip`: dataset
- `proyecto_final_ml.ipynb`: ingesta y ELT, EDA, modelo, predicciones y conclusiones
- `modelo_tamizaje.joblib`: modelo entrenado
- `proyecto_ml.db`: base de datos SQLite
- `app.py`: aplicación web
- `simulador_asd.html`: simulador en HTML
- `requirements.txt`: dependencias
- `README.md`: este archivo

## Metodología

1. Ingesta y limpieza de datos
2. Análisis exploratorio (EDA)
3. Entrenamiento del modelo de clasificación
4. Evaluación con métricas (accuracy, precisión, recall, F1)
5. Predicciones y conclusiones

## Resultados

Se usó el dataset real de Kaggle (1054 registros, 19 columnas, sin valores nulos ni duplicados). El 69.1% de los casos pertenece a la clase "con rasgos".

Se compararon tres modelos con validación cruzada y se evaluaron en un conjunto de prueba de 211 casos:

| Modelo | Accuracy | Precisión | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Regresión logística (elegido) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Gradient Boosting | 0.981 | 0.986 | 0.986 | 0.986 | 0.999 |
| Random Forest | 0.957 | 0.966 | 0.973 | 0.969 | 0.996 |

**Hallazgos:**

- Los ítems del cuestionario (A1 a A10) concentran el poder predictivo. La variable más importante fue A9.
- Con solo datos demográficos, el ROC-AUC baja a 0.570.

**Limitación importante:** en el dataset original la clase se deriva de la puntuación del cuestionario, por eso las métricas son casi perfectas. El modelo aprende a reproducir esa regla y no a diagnosticar. Es una herramienta de apoyo, no un diagnóstico clínico.

## Cómo ejecutarlo

1. Descarga el repositorio.
2. Descomprime `archive.zip`.
3. Abre el cuaderno en Jupyter o Google Colab y ejecuta las celdas en orden.

## Aplicación desplegada

https://ml-deteccion-tea-ninos-ugei4um3sv7kqmfvppemor.streamlit.app/

## Video de presentación

Enlace pendiente.
