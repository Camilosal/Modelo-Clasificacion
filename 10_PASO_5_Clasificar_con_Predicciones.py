import streamlit as st
import pandas as pd
from pathlib import Path
from utils import (
    ejecutar_script, get_active_topic, check_pipeline_status, 
    load_config, save_config, get_active_topic_config,
    get_classifier_model_path, get_predictions_path
)

st.set_page_config(
    page_title="Paso 5 Clasificar con Predicciones",
    page_icon="assets/logo.png"
)

# --- Constantes y Configuración ---
TOPIC_NAME = get_active_topic()

st.title(f"✨ PASO 5: Clasificar con Predicciones para '{TOPIC_NAME.capitalize()}'")
st.markdown("---")

# Verificar si hay un tema activo
if not TOPIC_NAME:
    st.error("No hay un tema activo seleccionado.")
    st.info("Por favor, ve a la página de 'Configuración' para seleccionar o crear un tema.")
    st.stop()

# Mostrar el estado actual del paso
pipeline_status = check_pipeline_status(TOPIC_NAME)
step_status = pipeline_status.get("Paso 5: Clasificar con Predicciones", {"estado": "Desconocido", "detalle": ""})
st.info(f"**Estado actual:** {step_status['estado']} - {step_status['detalle']}")
st.markdown("---")

st.markdown("""
¡El último paso! Aquí es donde todo el trabajo cobra sentido.
El sistema utilizará el modelo que entrenaste en el Paso 4 para analizar el archivo de datos **preprocesado (Parquet)** completo.
El resultado será un archivo Excel con todos tus contratos originales, más dos columnas nuevas: la predicción del modelo y su nivel de confianza.
""")

model_file = get_classifier_model_path(TOPIC_NAME)
predictions_file = get_predictions_path(TOPIC_NAME, format='csv')

config = load_config()
topic_config = config.get("TOPICS", {}).get(TOPIC_NAME, {})

model_ready = model_file.exists()

st.markdown("#### 2. Configura y Ejecuta la Clasificación")

# [NUEVO] Añadir control para el tamaño del lote de predicción
batch_size = st.number_input(
    "Tamaño del Lote de Procesamiento:",
    min_value=1000,
    max_value=200000,
    value=topic_config.get("BATCH_SIZE_PREDICTION", 100000),
    step=1000,
    help="Define cuántos registros procesar a la vez. Un número más bajo consume menos RAM pero puede tardar más. Un número más alto es más rápido pero requiere más RAM."
)

if st.button("🚀 Ejecutar Clasificación Masiva", disabled=not model_ready, type="primary"):
    # Guardar el tamaño del lote en la configuración antes de ejecutar
    config["TOPICS"][TOPIC_NAME]["BATCH_SIZE_PREDICTION"] = batch_size
    save_config(config)
    st.toast(f"Tamaño de lote guardado: {batch_size}", icon="💾")

    ejecutar_script("6_Ejecutar_Clasificador.py", show_progress_bar=True)
    st.info("✅ ¡Clasificación completada! El archivo de predicciones está listo para descargar.")
    st.rerun()

if not model_ready:
    st.warning("El modelo final aún no ha sido entrenado. Por favor, completa el 'Paso 4: Entrenar Clasificador' para poder continuar.")

if predictions_file.exists() and step_status['estado'] == 'Completado':
    st.markdown("---")
    st.header("Resultados de la Clasificación")
    st.success(f"El archivo de predicciones '{predictions_file.name}' ha sido generado exitosamente.")

    try:
        # Leer las columnas necesarias para los resúmenes
        prediction_col = f'Prediccion_{TOPIC_NAME.capitalize()}'
        confidence_col = f'Confianza_{TOPIC_NAME.capitalize()}_SI'
        df_pred_summary = pd.read_csv(predictions_file, usecols=[prediction_col, confidence_col])

        if not df_pred_summary.empty:
            st.subheader("📊 Resumen de Predicciones")
            counts = df_pred_summary[prediction_col].value_counts()
            
            si_count = int(counts.get('SI', 0))
            no_count = int(counts.get('NO', 0))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Clasificados", f"{(si_count + no_count):,}")
            col2.metric(f"Relevantes (SI)", f"{si_count:,}")
            col3.metric(f"No Relevantes (NO)", f"{no_count:,}")
            
            # Reordenar para que el gráfico sea consistente (SI, luego NO)
            counts_ordered = counts.reindex(['SI', 'NO']).fillna(0)
            st.bar_chart(counts_ordered)

            # --- Gráfico de Distribución de Confianza para 'SI' ---
            df_si = df_pred_summary[df_pred_summary[prediction_col] == 'SI']
            if not df_si.empty:
                st.subheader("Distribución de Confianza para Predicciones 'SI'")
                st.info("Este gráfico muestra cuántas predicciones 'SI' caen en diferentes rangos de confianza. Idealmente, la mayoría deberían agruparse en la parte alta (90-100%).")
                
                # Usar numpy para crear los bins del histograma
                import numpy as np
                hist_values, bin_edges = np.histogram(
                    df_si[confidence_col], 
                    bins=10, # 10 barras de 5% cada una, de 50% a 100%
                    range=(0.5, 1.0)
                )
                
                # Crear un DataFrame para el gráfico de barras
                bin_labels = [f"{int(edge*100)}-{int(bin_edges[i+1]*100)}%" for i, edge in enumerate(bin_edges[:-1])]
                hist_df = pd.DataFrame(hist_values, index=bin_labels, columns=["Número de Contratos"])
                
                st.bar_chart(hist_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"No se pudo generar el resumen general de predicciones: {e}")

    # --- Resumen por 'Tipo de Contrato' ---
    try:
        # Leer las columnas de predicción y 'Tipo de Contrato'
        prediction_col = f'Prediccion_{TOPIC_NAME.capitalize()}'
        type_col = 'Tipo de Contrato'  # Asegúrate de que esta es la columna correcta en tu archivo CSV
        df_pred_type = pd.read_csv(predictions_file, usecols=[prediction_col, type_col], dtype=str)

        if not df_pred_type.empty:
            st.subheader("📊 Resumen de Predicciones por Tipo de Contrato")

            # Agrupar por 'Tipo de Contrato' y contar las predicciones 'SI' y 'NO'
            summary_by_type = df_pred_type.groupby(type_col)[prediction_col].value_counts().unstack(fill_value=0)

            # Mostrar la tabla resumen
            st.dataframe(summary_by_type, use_container_width=True)

            # Opcional: Gráfico de barras apiladas para visualizar la distribución
            st.bar_chart(summary_by_type, use_container_width=True)

    except Exception as e:
        st.error(f"No se pudo generar el resumen por tipo de contrato: {e}")

    # --- Descarga de Archivos de Resultados ---
    st.markdown("### 📥 Descargar Reportes")

    # Botón para descargar el Excel con los 'SI'
    predictions_excel_si_file = results_path / f"predicciones_{TOPIC_NAME}_SI.xlsx"
    if predictions_excel_si_file.exists():
        with open(predictions_excel_si_file, "rb") as file:
            st.download_button(
                label="⬇️ Descargar Excel (Solo Relevantes 'SI')",
                data=file,
                file_name=predictions_excel_si_file.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Descarga el archivo Excel solo con los contratos clasificados como 'SI'."
            )

    # --- Mensaje de felicitación y ciclo de retroalimentación ---
    config = load_config()
    topic_config = config.get("TOPICS", {}).get(TOPIC_NAME, {})
    active_loop_completed = topic_config.get("active_loop_completed", False)

    if active_loop_completed:
        st.balloons()
        st.success("¡Felicitaciones! Has completado un ciclo de retroalimentación, por lo que la calidad de estos reportes es alta.")
    else:
        st.warning("🔒 Calidad de Reporte Intermedia", icon="⚠️")
        st.markdown("""
        Para asegurar la máxima calidad y fiabilidad del reporte final, se recomienda completar al menos un **Ciclo de Retroalimentación Activa**. 
        Este proceso refina el modelo con los casos más difíciles, mejorando significativamente su precisión.
        
        **Para mejorar la calidad, puedes:**
        1.  Usar la opción **'Iniciar Ciclo de Retroalimentación Activa'** más abajo.
        2.  Seguir los pasos de validación y re-entrenamiento.
        """)

    # --- Ciclo de Retroalimentación Activa ---
    st.markdown("---")
    with st.expander("🔁 Iniciar Ciclo de Retroalimentación Activa (Opcional Avanzado)"):
        st.markdown("""
        **¿Por qué iniciar este ciclo?**
        Para mejorar drásticamente la precisión del modelo. En lugar de validar datos al azar, este proceso se enfoca en los casos donde el modelo tuvo **más dudas** (baja confianza). Al corregir sus puntos débiles, el modelo aprende más rápido y se vuelve más robusto.

        **¿Cómo funciona el ciclo?**
        1.  **Generar Lista de Revisión:** Al presionar el botón de abajo, el sistema creará un nuevo archivo Excel (`revision_activa_{tema}.xlsx`) con los contratos más ambiguos para el modelo.
        2.  **Validar las Predicciones:** Ve a la página **📝 PASO 2: Validación Humana**.
        3.  **Corregir en la Pestaña Correcta:** Dentro del Paso 2, selecciona la pestaña **"Revisión Activa (2º Ciclo)"**. Allí encontrarás la nueva lista. Tu tarea es corregir las predicciones que el modelo hizo incorrectamente.
        4.  **Consolidar y Re-entrenar:** Una vez que termines de validar en el Paso 2, usa el botón "🤝 Consolidar Todas las Validaciones". Luego, regresa al **🏆 PASO 4: Entrenar Clasificador** para crear una nueva versión de tu modelo, ahora mucho más inteligente.
        """)
        if st.button("🔍 Generar Lista de Revisión Activa"):
            ejecutar_script("7_Generar_Revision_Desde_Predicciones.py")
            st.info("✅ Lista de revisión activa generada. Ahora ve al **📝 PASO 2: Validación Humana** para completar este segundo ciclo de revisión.")
            st.rerun()
