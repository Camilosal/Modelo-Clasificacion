import streamlit as st
from pathlib import Path
import pandas as pd
from utils import (
    ejecutar_script, get_active_topic, check_pipeline_status, 
    get_active_topic_config, get_keywords_file_path, get_preprocessed_data_path,
    load_config, save_config
)

st.set_page_config(
    page_title="Paso 1: Generar Candidatos",
    page_icon="🚀",
    layout="wide"
)

# --- Constantes y Configuración ---
BASE_DIR = Path(__file__).resolve().parent.parent
TOPIC_NAME = get_active_topic()
config = load_config()

# --- Interfaz de Usuario ---
st.title(f"🚀 PASO 1: Generar Candidatos para '{TOPIC_NAME.capitalize()}'")
st.markdown("---")

# Verificar si hay un tema activo
if not TOPIC_NAME:
    st.error("No hay un tema activo seleccionado.")
    st.info("Por favor, ve a la página de 'Configuración' para seleccionar o crear un tema.")
    st.stop()

# Mostrar el estado actual del paso
pipeline_status = check_pipeline_status(TOPIC_NAME)
step_status = pipeline_status.get("Paso 1: Generar Candidatos", {"estado": "Desconocido", "detalle": ""})
st.info(f"**Estado actual:** {step_status['estado']} - {step_status['detalle']}")
st.markdown("---")

st.markdown(f"""
Este es el punto de partida del pipeline. El sistema analizará el archivo de datos **preprocesado (Parquet)** y buscará contratos que podrían ser relevantes para tu tema: **{TOPIC_NAME}**.
Utiliza dos métodos de búsqueda:
1.  **Búsqueda por Keywords:** Encuentra coincidencias directas con las palabras clave que definiste en la página de Configuración.
2.  **Búsqueda Semántica:** Utiliza un modelo de IA para encontrar contratos que hablen de lo mismo, incluso si no usan las palabras exactas.
""")

# --- Verificaciones de Prerrequisitos ---
parquet_file = get_preprocessed_data_path(TOPIC_NAME)
keywords_file = get_keywords_file_path(TOPIC_NAME)

parquet_ready = parquet_file.exists()
keywords_ready = keywords_file.exists()

st.markdown("#### Requisitos para Ejecutar:")
if parquet_ready:
    st.success(f"✅ Archivo de datos preprocesado (`{parquet_file.name}`) encontrado.")
else:
    st.error(f"❌ Archivo de datos preprocesado no encontrado. Por favor, ve a la página de **⚙️ Configuración de Busqueda** y genera el archivo Parquet primero.")

if keywords_ready:
    st.success(f"✅ Archivo de keywords (`{keywords_file.name}`) encontrado.")
else:
    st.error(f"❌ Archivo de keywords (`{keywords_file.name}`) no encontrado. Por favor, créalo en la página de **⚙️ Configuración de Busqueda**.")

# --- Mostrar Configuración Actual (NUEVO) ---
if parquet_ready:
    st.markdown("--- ")
    st.markdown("#### Configuración de Búsqueda Actual")
    topic_config = get_active_topic_config()
    text_cols = topic_config.get("TEXT_COLUMNS_TO_COMBINE", [])
    unspsc_col = topic_config.get("FILTRADO_UNSPSC", {}).get("COLUMNA_UNSPSC", "No configurada")

    st.info(f"**Columnas de texto a analizar:** `{', '.join(text_cols)}`")
    st.info(f"**Columna de códigos UNSPSC:** `{unspsc_col}`")
    st.markdown("Puedes cambiar estos parámetros en el **🎛️ Panel de Control**.")

st.markdown("---")

# --- Botón de Ejecución ---
all_files_ready = parquet_ready and keywords_ready
if st.button("🚀 Ejecutar Búsqueda de Candidatos", disabled=not all_files_ready):
    # La función ejecutar_script ahora devuelve True si tiene éxito, False si falla.
    success = ejecutar_script("1_Seleccion_Candidatos.py", show_progress_bar=True)
    
    if success:
        st.success("🎉 ¡Proceso finalizado con éxito!")
        st.info("**Siguiente Paso:** Ve a la página **📝 PASO 2: Validación Humana** en el menú de la izquierda para revisar los candidatos generados.")
        st.info("Puedes ver el número de candidatos encontrados en los logs de ejecución de arriba.")
        st.rerun()
