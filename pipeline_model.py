from math import isfinite
from typing import Any

import pandas as pd


PREDICTOR_COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
]

ZERO_AS_MISSING = [
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
]

INTEGER_COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "age",
]


def ceros_clinicos_a_nan(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned[ZERO_AS_MISSING] = cleaned[ZERO_AS_MISSING].replace(0, float("nan"))
    return cleaned


def validar_entrada(values: dict[str, Any]) -> dict[str, float]:
    missing = [column for column in PREDICTOR_COLUMNS if column not in values or values[column] is None]
    if missing:
        raise ValueError("Completa todos los campos antes de ejecutar la predicción.")

    record: dict[str, float] = {}
    for column in PREDICTOR_COLUMNS:
        try:
            value = float(values[column])
        except (TypeError, ValueError) as error:
            raise ValueError(f"El valor de {column} debe ser numérico.") from error
        if not isfinite(value) or value < 0:
            raise ValueError(f"El valor de {column} debe ser finito y no negativo.")
        if column in INTEGER_COLUMNS and not value.is_integer():
            raise ValueError(f"El valor de {column} debe ser un número entero.")
        record[column] = value
    return record


def predecir_registro(values: dict[str, Any], model: Any) -> tuple[int, float]:
    record = validar_entrada(values)
    sample = pd.DataFrame([record], columns=PREDICTOR_COLUMNS)
    prediction = int(model.predict(sample)[0])
    class_probabilities = dict(zip(model.classes_, model.predict_proba(sample)[0]))
    return prediction, float(class_probabilities[1])
