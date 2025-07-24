import streamlit as st
import pandas as pd
from pathlib import Path
import io
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from utils import (
    ejecutar_script, get_active_topic, check_pipeline_status, 
    load_keywords, save_keywords, get_active_topic_config,
    get_human_review_file_path, get_report_path, get_finetuning_dataset_path,
    get_keywords_file_path
)

st.set_page_config(
    page_title="Paso 3: Aprender y Refinar",
    page_icon="assets/logo.png"
)

# --- Constantes y Configuración ---
TOPIC_NAME = get_active_topic()

st.title(f"PASO 3: Aprender y Refinar para '{TOPIC_NAME.capitalize()}'")
st.markdown("---")

# Verificar si hay un tema activo
if not TOPIC_NAME:
    st.error("No hay un tema activo seleccionado.")
    st.info("Por favor, ve a la página de 'Configuración' para seleccionar o crear un tema.")
    st.stop()

# Mostrar el estado actual del paso
pipeline_status = check_pipeline_status(TOPIC_NAME)
step_status = pipeline_status.get("Paso 3: Aprender y Refinar", {"estado": "Desconocido", "detalle": ""})
st.info(f"**Estado actual:** {step_status['estado']} - {step_status['detalle']}")
st.markdown("---")

st.markdown("""
En este paso, utilizas los datos que validaste en el Paso 2 para hacer que el sistema sea más inteligente. Tienes dos opciones principales para refinar el proceso.
""")
st.info("**Antes de empezar, asegúrate de haber validado una cantidad significativa de contratos en el Paso 2.** Mientras más datos validados, mejores serán los resultados de este paso.")

review_file = get_human_review_file_path(TOPIC_NAME)
report_file = get_report_path(TOPIC_NAME, 'rendimiento_keywords')
finetuning_dataset_file = get_finetuning_dataset_path(TOPIC_NAME)

validation_ready = review_file.exists()

st.markdown("### Opción 1: Analizar Rendimiento de Keywords")
st.markdown("""
**¿Qué hace?** Genera un reporte en Excel que te muestra qué tan efectivas son tus palabras clave y temas. Podrás ver cuáles keywords atraen contratos correctos (alta precisión) y cuáles atraen "ruido".
**¿Para qué sirve?** Para mejorar tu archivo de `keywords.xlsx` en la página de Inicio, eliminando términos poco efectivos y potenciando los que sí funcionan.
""")
if st.button("📊 Generar Reporte de Keywords", disabled=not validation_ready):
    st.info("Generando reporte de rendimiento de keywords...")
    success_report = ejecutar_script("2_Analisis_Feedback_Keywords.py")
    
    if success_report:
        st.success("✅ Reporte de keywords generado.")
        st.info("Actualizando automáticamente el dataset de entrenamiento con las últimas validaciones...")
        
        success_dataset = ejecutar_script("3_Generar_Datos_FineTuning.py")
        
        if success_dataset:
            # Borrar el estado para forzar una recarga desde el archivo en la pestaña de edición
            if 'df_finetune' in st.session_state:
                del st.session_state.df_finetune
            st.success("✅ ¡Proceso completado! El reporte está listo y el dataset de entrenamiento ha sido actualizado.")
        else:
            st.error("Falló la actualización automática del dataset de entrenamiento. El reporte de keywords sí fue generado.")

# --- Visualización del Reporte de Keywords ---
if report_file.exists():
    with open(report_file, "rb") as file:
        st.download_button(
            label="📥 Descargar Reporte Completo en Excel",
            data=file,
            file_name=report_file.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.markdown("#### Visualización del Reporte")
    tab_table, tab_charts, tab_suggestions = st.tabs(["📄 Tabla Detallada", "📊 Gráficos Resumen", "💡 Sugerencias de Keywords"])

    with tab_table:
        st.markdown("Analiza la precisión de cada palabra clave y tema. Ordena la tabla haciendo clic en los encabezados para identificar los términos más y menos efectivos.")
        try:
            df_keywords_table = pd.read_excel(report_file, sheet_name='Rendimiento por Keyword')
            df_themes_table = pd.read_excel(report_file, sheet_name='Rendimiento por Tema')

            st.markdown("##### Rendimiento por Palabra Clave")
            st.dataframe(df_keywords_table.style.format({"Precision": "{:.2%}"}), use_container_width=True)

            st.markdown("##### Rendimiento por Tema")
            st.dataframe(df_themes_table.style.format({"Precision": "{:.2%}"}), use_container_width=True)

        except Exception as e:
            st.info("Genera un reporte para ver la tabla de rendimiento detallada aquí.")

    with tab_charts:
        st.markdown("Visualiza rápidamente las 20 palabras clave y temas con mayor precisión. Esto te ayuda a identificar rápidamente los términos más valiosos.")
        try:
            df_keywords_chart = pd.read_excel(report_file, sheet_name='Rendimiento por Keyword', index_col=0)
            df_themes_chart = pd.read_excel(report_file, sheet_name='Rendimiento por Tema', index_col=0)

            st.markdown("##### Top 20 Palabras Clave por Precisión")
            top_keywords = df_keywords_chart.head(20)
            st.bar_chart(top_keywords[['Precision']], use_container_width=True)

            st.markdown("##### Top 20 Temas por Precisión")
            top_themes = df_themes_chart.head(20)
            st.bar_chart(top_themes[['Precision']], use_container_width=True)

        except Exception as e:
            st.info("Genera un reporte para ver los gráficos de rendimiento aquí.")

    with tab_suggestions:
        st.markdown("##### Nuevas Keywords Sugeridas por el Sistema")
        st.info("""
        Aquí se muestran las palabras más frecuentes encontradas en los contratos que validaste como **'SI'**, pero que **no fueron detectados por tus keywords actuales**.
        
        Estas son excelentes candidatas para añadir a tu lista de keywords en la página de **🏠 Inicio** para mejorar la cobertura en el próximo ciclo.
        """)
        try:
            df_sugerencias = pd.read_excel(report_file, sheet_name='Sugerencias Keywords')
            if not df_sugerencias.empty:
                st.markdown("---")
                st.markdown("##### Acción Rápida: Añadir Sugerencias a tu Lista")
                
                # Añadir columna de selección para el editor de datos
                df_sugerencias['Seleccionar'] = False
                
                # Usar st.data_editor para una tabla interactiva
                edited_df_sugerencias = st.data_editor(
                    df_sugerencias,
                    column_config={
                        "Seleccionar": st.column_config.CheckboxColumn("Seleccionar", required=True),
                        "Keyword Sugerida": st.column_config.TextColumn("Keyword Sugerida", disabled=True),
                        "Apariciones": st.column_config.NumberColumn("Apariciones", disabled=True) # Cambiar "Frecuencia" por "Apariciones"
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                selected_keywords = edited_df_sugerencias[edited_df_sugerencias['Seleccionar']]['Keyword Sugerida'].tolist()

                # Cargar los temas existentes para el selectbox
                keywords_file = BASE_DIR / f"keywords_{TOPIC_NAME}.xlsx"
                existing_keywords_data = load_keywords(keywords_file)
                existing_themes = list(existing_keywords_data.keys())

                # UI para seleccionar un tema
                col1, col2 = st.columns([2, 1])
                with col1:
                    theme_option = st.selectbox(
                        "Selecciona un tema existente o crea uno nuevo",
                        options=existing_themes + ["➕ Crear nuevo tema..."],
                        index=None,
                        placeholder="Elige un tema..."
                    )

                new_theme_name = ""
                if theme_option == "➕ Crear nuevo tema...":
                    with col2:
                        new_theme_name = st.text_input("Nombre del nuevo tema:", label_visibility="collapsed", placeholder="Nombre del nuevo tema...")

                target_theme = new_theme_name.strip().replace(" ", "_") if new_theme_name else theme_option

                if st.button("➕ Añadir Keywords Seleccionadas al Tema", disabled=not selected_keywords or not target_theme):
                    current_kws_for_theme = existing_keywords_data.get(target_theme, [])
                    updated_kws = sorted(list(set(current_kws_for_theme + selected_keywords)))
                    existing_keywords_data[target_theme] = updated_kws
                    save_keywords(keywords_file, existing_keywords_data)
                    st.success(f"¡Se añadieron {len(selected_keywords)} keywords al tema '{target_theme}'! La lista se ha actualizado en la página de **🏠 Inicio**.")
        except FileNotFoundError:
            st.write("La versión actual de tu reporte no incluye sugerencias. Vuelve a generar el reporte para ver esta sección.")

st.markdown("---")
st.markdown("### Opción 2: Crear un Modelo Semántico Experto (Avanzado)")
st.markdown("""
**¿Qué hace?** Entrena un modelo de lenguaje profundo (fine-tuning) para que se convierta en un experto en la jerga y el contexto de tu tema específico.
**¿Para qué sirve?** Para mejorar drásticamente la "Búsqueda Semántica" del Paso 1. Un modelo experto encontrará candidatos mucho más precisos y relevantes que un modelo genérico.
""")
st.warning("**¿Cuándo usar esto?** Cuando la búsqueda por keywords no es suficiente o cuando quieres llevar la precisión del sistema al siguiente nivel. Requiere una buena cantidad de datos validados (tanto 'SI' como 'NO').")

# --- A. Gestionar Datos de Entrenamiento ---
st.markdown("#### A. Gestionar Datos de Entrenamiento")
st.markdown("El primer paso para crear un modelo experto es preparar su material de estudio: un **dataset de entrenamiento**. Este dataset consiste en pares de textos de contratos con una etiqueta que indica si son similares (score 1.0) o diferentes (score 0.0).")

tab_gen, tab_edit, tab_updown = st.tabs(["🤖 Generar Automáticamente", "✍️ Editar Interactivamente", "📤 Cargar/Descargar Archivo"])

with tab_gen:
    st.markdown("Usa tus validaciones ('SI' vs 'NO') para crear automáticamente un conjunto de datos de entrenamiento. El sistema creará pares de textos 'SI' vs 'SI' (similares) y 'SI' vs 'NO' (diferentes).")
    st.info("Esta es la forma recomendada de empezar.")
    if st.button("⚙️ Generar/Regenerar Dataset desde Validaciones", disabled=not validation_ready):
        if ejecutar_script("3_Generar_Datos_FineTuning.py"):
            # Después de la generación, borramos el estado para forzar una recarga desde el archivo
            if 'df_finetune' in st.session_state:
                del st.session_state.df_finetune
            st.success("✅ Dataset generado. Ahora puedes revisarlo en la pestaña '✍️ Editar Interactivamente' o proceder a entrenar el modelo.")
            st.rerun()

# Cargar el dataset en el estado de la sesión si existe
if finetuning_dataset_file.exists():
    if 'df_finetune' not in st.session_state or st.session_state.get('topic_name_for_finetune') != TOPIC_NAME:
        try:
            st.session_state.df_finetune = pd.read_csv(finetuning_dataset_file)
            st.session_state.topic_name_for_finetune = TOPIC_NAME
        except Exception as e:
            st.error(f"Error al cargar el dataset de fine-tuning: {e}")

with tab_edit:
    if 'df_finetune' in st.session_state:
        st.markdown("Aquí puedes editar, añadir o eliminar manualmente pares de frases. Es útil para inyectar conocimiento experto específico que no se captura en la generación automática. **Guarda tus cambios antes de entrenar el modelo.**")
        edited_df_finetune = st.data_editor(
            st.session_state.df_finetune,
            num_rows="dynamic",
            use_container_width=True,
            height=400,
            column_config={
                "score": st.column_config.NumberColumn(
                    "Similitud (Score)",
                    help="1.0 para frases que deben ser similares, 0.0 para diferentes.",
                    min_value=0.0, max_value=1.0, step=1.0, format="%.1f"
                )
            },
            key=f"finetune_editor_{TOPIC_NAME}"
        )

        if st.button("💾 Guardar Cambios del Editor", key="save_finetune_editor"):
            if 'score' not in edited_df_finetune.columns or not all(edited_df_finetune['score'].isin([0.0, 1.0])):
                 st.error("La columna 'score' solo puede contener los valores 0.0 o 1.0.")
            else:
                edited_df_finetune.to_csv(finetuning_dataset_file, index=False, encoding='utf-8-sig')
                st.session_state.df_finetune = edited_df_finetune
                st.success("¡Dataset de fine-tuning guardado exitosamente!")
                st.info("✅ Dataset guardado. Ya puedes proceder a entrenar el modelo experto en la sección B.")
    else:
        st.info("Primero genera un dataset usando la pestaña 'Generar Automáticamente' o carga un archivo en 'Cargar/Descargar'.")

with tab_updown:
    st.markdown("Usa esta opción para descargar el dataset, editarlo masivamente en Excel o compartirlo con otros expertos. Luego, puedes subir la versión modificada para actualizar el sistema.")
    df_to_download_finetune = st.session_state.get('df_finetune', pd.DataFrame())
    if not df_to_download_finetune.empty:
        csv_data_finetune = df_to_download_finetune.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button("📥 Descargar dataset actual", data=csv_data_finetune, file_name=finetuning_dataset_file.name, mime="text/csv")

    st.markdown("---")
    uploaded_finetune_file = st.file_uploader("📂 Sube aquí tu dataset de fine-tuning (.csv)", type=['csv'], key="finetune_uploader")
    if uploaded_finetune_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_finetune_file)
            if not {'frase1', 'frase2', 'score'}.issubset(df_uploaded.columns):
                st.error("El archivo CSV debe contener las columnas 'frase1', 'frase2', y 'score'.")
            else:
                df_uploaded.to_csv(finetuning_dataset_file, index=False, encoding='utf-8-sig')
                st.session_state.df_finetune = df_uploaded
                st.success(f"✅ Archivo '{uploaded_finetune_file.name}' cargado. La tabla de 'Editar Interactivamente' ha sido actualizada.")
                st.rerun()
        except Exception as e:
            st.error(f"No se pudo leer o guardar el archivo subido: {e}")

# --- B. Entrenar Modelo Experto ---
st.markdown("---")
st.markdown("#### B. Entrenar Modelo Experto")
st.markdown("Esta acción alimenta al modelo de IA base con tus ejemplos de entrenamiento para crear una versión especializada (fine-tuning). El modelo aprenderá la jerga y el contexto de tu tema, mejorando drásticamente su capacidad para encontrar contratos relevantes por similitud semántica.")
st.markdown("Una vez que estés satisfecho con tu dataset, ejecuta este proceso. **Al terminar, el sistema iniciará automáticamente el Paso 1 para re-procesar los datos con el nuevo modelo experto.**")

if st.button("🧠 Entrenar Modelo Experto", disabled=not finetuning_dataset_file.exists()):
    st.markdown("### Parte 1: Entrenando Modelo Experto...")
    # Ejecutar el fine-tuning y mostrar su progreso
    entrenamiento_exitoso = ejecutar_script("4_Entrenar_Modelo_Preclasificacion.py", show_progress_bar=True)
    
    if entrenamiento_exitoso:
        st.success("✅ ¡Modelo experto entrenado con éxito!")
        st.markdown("---")
        st.markdown("### Parte 2: Re-procesando datos con el nuevo modelo...")
        st.info("Iniciando automáticamente el Paso 1 para generar una nueva lista de candidatos de alta calidad.")
        
        # Ejecutar el preprocesamiento y mostrar su progreso
        reprocesamiento_exitoso = ejecutar_script("1_Generar_Candidatos.py", show_progress_bar=True)
        
        if reprocesamiento_exitoso:
            st.balloons()
            st.success("¡Ciclo completo finalizado!")
            st.info("**Siguiente Paso:** Ahora puedes ir al **📝 PASO 2: Validación Humana** para revisar la nueva lista de candidatos generada por tu modelo experto.")
        else:
            st.error("El re-procesamiento (Paso 1) falló. Revisa los logs para más detalles.")
    else:
        st.error("El entrenamiento del modelo experto (Paso 5) falló. Revisa los logs para más detalles.")

st.caption("Al usar esta opción, el Paso 1 se ejecutará automáticamente después del entrenamiento.")

if not validation_ready:
    st.warning("Debes completar el Paso 2 (Validación Humana) para poder continuar con el aprendizaje.")
