from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import streamlit as st

from pipeline_model import PREDICTOR_COLUMNS, predecir_registro


MODEL_PATH = Path(__file__).with_name("modelo_diabetes.joblib")
LABELS = {
    "pregnancies": ("Embarazos", True),
    "glucose": ("Glucosa", True),
    "blood_pressure": ("Presión arterial diastólica", True),
    "skin_thickness": ("Grosor del pliegue cutáneo", True),
    "insulin": ("Insulina", True),
    "bmi": ("Índice de masa corporal (BMI)", False),
    "diabetes_pedigree_function": ("Función de pedigrí de diabetes", False),
    "age": ("Edad", True),
}

@st.cache_resource
def cargar_modelo() -> Any:
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Predicción académica de diabetes", page_icon="🩺")
st.title("Clasificación académica de registros de diabetes")
st.write(
    "Prototipo educativo basado en ocho variables clínicas y demográficas. "
    "La etiqueta estimada no es un diagnóstico."
)
st.warning(
    "Uso exclusivamente académico. No sustituye una valoración profesional ni debe "
    "utilizarse para tomar decisiones clínicas."
)
st.caption(
    "En este conjunto de datos, el valor 0 de glucosa, presión arterial, grosor cutáneo, "
    "insulina o BMI se interpreta como medición ausente y el pipeline lo imputa."
)

if not MODEL_PATH.is_file():
    st.error("No se encuentra modelo_diabetes.joblib. Ejecuta el notebook final para generarlo.")
    st.stop()

model = cargar_modelo()
with st.form("entrada_modelo"):
    values: dict[str, Any] = {}
    for column in PREDICTOR_COLUMNS:
        label, integer = LABELS[column]
        step = 1 if integer else 0.001 if column == "diabetes_pedigree_function" else 0.1
        number_format = "%d" if integer else "%.3f" if column == "diabetes_pedigree_function" else "%.2f"
        values[column] = st.number_input(
            label,
            min_value=0 if integer else 0.0,
            value=None,
            step=step,
            format=number_format,
            key=column,
        )
    submitted = st.form_submit_button("Evaluar registro")

if submitted:
    try:
        prediction, probability = predecir_registro(values, model)
    except ValueError as error:
        st.error(str(error))
    else:
        st.metric("Etiqueta estimada", prediction)
        st.write(f"Probabilidad estimada para la etiqueta 1: {probability:.1%}")
        st.info("La etiqueta reproduce patrones del conjunto de datos; no confirma ni descarta diabetes.")
