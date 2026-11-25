from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import streamlit as st


NOTEBOOK_PATH = Path(__file__).with_name("Proyectofinal.ipynb")


def _text(value: str | list[str]) -> str:
    return "".join(value) if isinstance(value, list) else value


@st.cache_data(show_spinner=False)
def cargar_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def mostrar_salida(output: dict[str, Any]) -> None:
    output_type = output.get("output_type")

    if output_type == "stream":
        text = _text(output.get("text", ""))
        if text:
            st.text(text)
        return

    if output_type == "error":
        st.error(f"{output.get('ename', 'Error')}: {output.get('evalue', '')}")
        return

    data = output.get("data", {})
    if "image/png" in data:
        st.image(base64.b64decode(_text(data["image/png"])), width="stretch")
    elif "image/jpeg" in data:
        st.image(base64.b64decode(_text(data["image/jpeg"])), width="stretch")
    elif "text/html" in data:
        st.html(_text(data["text/html"]))
    elif "text/markdown" in data:
        st.markdown(_text(data["text/markdown"]))
    elif "text/plain" in data:
        st.text(_text(data["text/plain"]))


st.set_page_config(page_title="Proyecto final · Minería de Datos", layout="wide")

if not NOTEBOOK_PATH.is_file():
    st.error(f"No se encuentra el notebook: {NOTEBOOK_PATH.name}")
    st.stop()

notebook = cargar_notebook()
st.title("Proyecto final · Minería de Datos")
st.caption("Presentación del notebook ejecutado, con sus resultados y gráficas.")

code_number = 0
for cell in notebook.get("cells", []):
    source = _text(cell.get("source", []))

    if cell.get("cell_type") == "markdown":
        if source.strip():
            st.markdown(source)
        continue

    if cell.get("cell_type") != "code":
        continue

    code_number += 1
    if source.strip():
        with st.expander(f"Código · celda {code_number}"):
            st.code(source, language="python")

    for output in cell.get("outputs", []):
        mostrar_salida(output)
