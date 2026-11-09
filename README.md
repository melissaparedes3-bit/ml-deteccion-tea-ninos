
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
