import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEMILLA = 42

st.set_page_config(page_title="Detección temprana de rasgos TEA", page_icon="🧩", layout="centered")

PREGUNTAS = [
    "A1. ¿Mira al adulto cuando se le llama por su nombre?",
    "A2. ¿Es fácil lograr contacto visual con él/ella?",
    "A3. ¿Señala con el dedo para pedir algo que quiere?",
    "A4. ¿Señala con el dedo para compartir algo que le interesa?",
    "A5. ¿Juega a imaginar (cuidar muñecas, hablar por teléfono de juguete)?",
    "A6. ¿Sigue con la mirada hacia donde usted mira?",
    "A7. Si alguien de la familia está triste, ¿muestra ganas de consolarlo?",
    "A8. ¿Sus primeras palabras fueron típicas para su edad?",
    "A9. ¿Usa gestos simples (decir adiós con la mano)?",
    "A10. ¿Se queda mirando al vacío sin una razón aparente?",
]
# En A1-A9 la respuesta "rara vez / nunca" suma punto de riesgo; en A10 suma "a menudo / siempre".
# El dataset ya trae cada ítem codificado como 0/1 (1 = respuesta de riesgo).

DATA_DIR = Path(__file__).parent


@st.cache_data
def cargar_datos():
    csv = next(DATA_DIR.glob("*.csv"), None)
    if csv is not None:
        df = pd.read_csv(csv)
    else:
        with zipfile.ZipFile(DATA_DIR / "archive.zip") as z:
            nombre = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            with z.open(nombre) as f:
                df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    return df


def preparar(df):
    X = df[[f"A{i}" for i in range(1, 11)] + ["Age_Mons"]].copy()
    X["Sex"] = (df["Sex"] == "m").astype(int)
    X["Jaundice"] = (df["Jaundice"] == "yes").astype(int)
    X["Family_mem_with_ASD"] = (df["Family_mem_with_ASD"] == "yes").astype(int)
    y = (df["Class/ASD Traits"] == "Yes").astype(int)
    return X, y


@st.cache_resource
def entrenar():
    df = cargar_datos()
    X, y = preparar(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEMILLA, stratify=y)
    clasificadores = {
        "Regresión logística": LogisticRegression(max_iter=1000, random_state=SEMILLA),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEMILLA),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEMILLA),
    }
    metricas, ajustados = [], {}
    for nombre, clf in clasificadores.items():
        # La edad se estandariza; el resto de columnas pasa sin cambios (igual que en el cuaderno)
        pipe = Pipeline(
            [
                ("prep", ColumnTransformer([("num", StandardScaler(), ["Age_Mons"])], remainder="passthrough")),
                ("clf", clf),
            ]
        )
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        prob = pipe.predict_proba(Xte)[:, 1]
        metricas.append(
            {
                "Modelo": nombre,
                "Exactitud": round(accuracy_score(yte, p), 3),
                "Precisión": round(precision_score(yte, p), 3),
                "Recall": round(recall_score(yte, p), 3),
                "F1": round(f1_score(yte, p), 3),
                "ROC-AUC": round(roc_auc_score(yte, prob), 3),
            }
        )
        ajustados[nombre] = (pipe, confusion_matrix(yte, p))
    return ajustados, pd.DataFrame(metricas), X.columns.tolist(), len(df)


modelos, tabla_metricas, columnas, n_registros = entrenar()

st.title("🧩 Detección temprana de rasgos del espectro autista")
st.caption("Proyecto final de Machine Learning · Maestría en IA Aplicada a la Educación e Investigación")

st.warning(
    "Esta herramienta es **educativa**. No es un diagnóstico médico. "
    "Ante cualquier duda sobre el desarrollo de un niño, consulte a un profesional de la salud."
)

tab1, tab2, tab3 = st.tabs(["Simulador", "Modelos y métricas", "Sobre el proyecto"])

with tab1:
    st.subheader("Cuestionario de tamizaje (tipo Q-Chat-10)")
    respuestas = []
    for i, texto in enumerate(PREGUNTAS):
        opciones = ["No (0)", "Sí, respuesta de riesgo (1)"]
        r = st.radio(texto, opciones, horizontal=True, key=f"q{i}")
        respuestas.append(1 if r.endswith("(1)") else 0)

    c1, c2, c3 = st.columns(3)
    edad = c1.slider("Edad (meses)", 12, 36, 24)
    sexo = c2.selectbox("Sexo", ["Femenino", "Masculino"])
    ictericia = c3.selectbox("Ictericia al nacer", ["No", "Sí"])
    familiar = st.selectbox("¿Familiar con TEA?", ["No", "Sí"])
    nombre_modelo = st.selectbox("Modelo", list(modelos.keys()))

    if st.button("Evaluar", type="primary"):
        fila = pd.DataFrame(
            [
                respuestas
                + [edad, int(sexo == "Masculino"), int(ictericia == "Sí"), int(familiar == "Sí")]
            ],
            columns=columnas,
        )
        modelo = modelos[nombre_modelo][0]
        prob = modelo.predict_proba(fila)[0][1]
        puntaje = sum(respuestas)
        st.metric("Puntaje Q-Chat-10", f"{puntaje} / 10")
        st.progress(float(prob), text=f"Probabilidad estimada de rasgos TEA: {prob:.0%}")
        if prob >= 0.5:
            st.error("El modelo sugiere **presencia de rasgos TEA**. Se recomienda una evaluación profesional.")
        else:
            st.success("El modelo **no sugiere** rasgos TEA con estas respuestas. Si hay inquietudes, consulte igualmente.")

with tab2:
    st.subheader("Comparación de modelos (20 % de datos para prueba)")
    st.dataframe(tabla_metricas, hide_index=True, width="stretch")
    for nombre, (_, cm) in modelos.items():
        st.markdown(f"**Matriz de confusión – {nombre}**")
        st.dataframe(
            pd.DataFrame(cm, index=["Real: No", "Real: Sí"], columns=["Pred: No", "Pred: Sí"]),
            width="stretch",
        )
    st.info(
        "En este dataset la etiqueta se asigna a partir del puntaje Q-Chat-10 (más de 3 puntos = «Sí»), "
        "por eso los modelos alcanzan métricas casi perfectas: aprenden esa regla. "
        "No debe interpretarse como capacidad de diagnosticar."
    )

with tab3:
    st.markdown(
        f"""
**Objetivo:** apoyar la detección temprana de rasgos del espectro autista en niños pequeños
mediante clasificación con aprendizaje automático.

**Datos:** dataset público de tamizaje *Autism Screening Data for Toddlers* (Kaggle),
con {n_registros:,} registros de niños de 12 a 36 meses.

**Variables del modelo:** las 10 preguntas A1–A10, edad en meses, sexo, ictericia al nacer y
familiar con TEA. Se excluyó el puntaje Q-Chat-10 para evitar redundancia (es la suma de A1–A10).

**Modelos:** regresión logística, Random Forest y Gradient Boosting, con partición 80/20 estratificada.

**Limitaciones:** el dataset es de tamizaje, no de diagnóstico clínico, y la etiqueta depende
directamente del puntaje del cuestionario.
"""
    )
