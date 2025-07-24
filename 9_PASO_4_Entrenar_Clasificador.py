import streamlit as st
import pandas as pd
import json
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from utils import (
    ejecutar_script, get_active_topic, check_pipeline_status, 
    load_config, save_config, get_active_topic_config,
    get_human_review_file_path, get_classifier_model_path, get_report_path
)

st.set_page_config(
    page_title="Paso 4: Entrenar Clasificador",
    page_icon="assets/logo.png"
)

# --- Constantes y Configuración ---
TOPIC_NAME = get_active_topic()

st.title(f"🏆 PASO 4: Entrenar Clasificador Final para '{TOPIC_NAME.capitalize()}'")
st.markdown("---")

# Verificar si hay un tema activo
if not TOPIC_NAME:
    st.error("No hay un tema activo seleccionado.")
    st.info("Por favor, ve a la página de 'Configuración' para seleccionar o crear un tema.")
    st.stop()

# Mostrar el estado actual del paso
pipeline_status = check_pipeline_status(TOPIC_NAME)
step_status = pipeline_status.get("Paso 4: Entrenar Clasificador", {"estado": "Desconocido", "detalle": ""})
st.info(f"**Estado actual:** {step_status['estado']} - {step_status['detalle']}")
st.markdown("---")

st.markdown("""
Este es el momento de la verdad. Aquí, el sistema tomará **todas las validaciones** que has realizado (tanto del ciclo principal como del ciclo activo) y las usará para entrenar un modelo de clasificación final.

Este modelo aprenderá a imitar tus decisiones y podrá ser utilizado en el siguiente paso para clasificar miles de contratos de forma masiva.
""")

validated_file = get_human_review_file_path(TOPIC_NAME)
model_file = get_classifier_model_path(TOPIC_NAME)
metrics_file = get_report_path(TOPIC_NAME, 'clasificacion').with_suffix('.json')

config = load_config()
topic_config = config.get("TOPICS", {}).get(TOPIC_NAME, {})

validation_ready = validated_file.exists()

st.markdown("#### 1. Selecciona el Modelo de Clasificación")
st.info("Elige el algoritmo que el sistema usará para aprender de tus validaciones. Puedes experimentar para ver cuál ofrece mejores resultados para tu tema.")

# --- Mapeo de nombres técnicos a nombres amigables ---
model_options_map = {
    "RandomForestClassifier": "Random Forest (Recomendado, versátil)",
    "LogisticRegression": "Regresión Logística (Rápido y bueno para textos)",
    "SVC": "Máquina de Soporte Vectorial (Potente, puede ser más lento)"
}

# Obtener la lista de nombres amigables para el selectbox
friendly_model_names = list(model_options_map.values())

# Obtener el modelo guardado y encontrar su índice en la lista de nombres amigables
saved_model_technical_name = topic_config.get("CLASSIFIER_MODEL", "RandomForestClassifier")
saved_model_friendly_name = model_options_map.get(saved_model_technical_name, friendly_model_names[0])
try:
    default_index = friendly_model_names.index(saved_model_friendly_name)
except ValueError:
    default_index = 0

selected_friendly_name = st.selectbox(
    "Algoritmo de Clasificación:",
    options=friendly_model_names,
    index=default_index,
    help="Random Forest es un buen punto de partida. La Regresión Logística suele ser muy eficiente para clasificación de texto."
)

st.markdown("#### 2. Ejecuta el Entrenamiento")
if st.button("🚀 Entrenar Clasificador", disabled=not validation_ready, type="primary"):
    # Encontrar el nombre técnico correspondiente al nombre amigable seleccionado
    selected_technical_name = [tech_name for tech_name, friendly_name in model_options_map.items() if friendly_name == selected_friendly_name][0]
    
    # Guardar la selección en el config.json antes de ejecutar
    topic_config["CLASSIFIER_MODEL"] = selected_technical_name
    save_config(config)
    st.toast(f"Modelo seleccionado: {selected_friendly_name}. Guardando y iniciando entrenamiento...", icon="🧠")

    ejecutar_script("5_Entrenamiento_Clasificador.py", show_progress_bar=True)
    st.info("✅ ¡Entrenamiento completado! Revisa las métricas de rendimiento a continuación.")
    st.rerun()

if not validation_ready:
    st.warning("Aún no existen datos validados. Por favor, completa los Pasos 1 y 2 para poder entrenar el modelo.")

# --- Visualización de Resultados del Entrenamiento ---
if metrics_file.exists() and step_status['estado'] == 'Completado':
    st.markdown("---")
    st.header("📊 Resultados del Entrenamiento")
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    st.success(f"Modelo **{model_options_map.get(topic_config.get('CLASSIFIER_MODEL'))}** entrenado y guardado.")

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Precisión Global (Accuracy)", f"{metrics['accuracy']:.2%}")
    col2.metric("✅ Precisión para 'SI'", f"{metrics.get('SI', {}).get('precision', 0):.2%}", help="De todo lo que el modelo clasificó como 'SI', ¿qué porcentaje acertó?")
    col3.metric("📈 Cobertura para 'SI' (Recall)", f"{metrics.get('SI', {}).get('recall', 0):.2%}", help="De todos los 'SI' reales, ¿qué porcentaje encontró el modelo?")

    st.markdown("#### Matriz de Confusión")
    st.info("Esta matriz muestra los aciertos y errores del modelo. Idealmente, los números en la diagonal (Verdaderos Positivos y Verdaderos Negativos) deben ser altos.")
    
    try:
        cm = metrics['confusion_matrix']
        df_cm = pd.DataFrame(cm, index=['Real: NO', 'Real: SI'], columns=['Predicción: NO', 'Predicción: SI'])
        
        fig, ax = plt.subplots()
        sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues', ax=ax)
        ax.set_title('Matriz de Confusión')
        ax.set_ylabel('Etiqueta Real')
        ax.set_xlabel('Etiqueta Predicha por el Modelo')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"No se pudo generar la matriz de confusión: {e}")

    st.markdown("#### Reporte Detallado")
    report_df = pd.DataFrame(metrics).transpose()
    st.dataframe(report_df)

    st.markdown("---")
    st.info("🎉 ¡Excelente! El modelo está listo. Ahora puedes ir al **✨ PASO 5: Clasificar con Predicciones** para usarlo en todo tu conjunto de datos.")