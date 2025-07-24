import streamlit as st
import pandas as pd
import json
from pathlib import Path
from utils import (
    time,
    get_active_topic, check_pipeline_status, ejecutar_script, 
    load_config, save_config, get_topic_results_dir, get_active_topic_config,
    get_human_review_file_path, get_report_path, get_predictions_path, get_preprocessed_data_path,
    get_file_columns
)

st.set_page_config(
    page_title="Panel de Control - Sistema de Clasificación",
    page_icon="🎛️",
    layout="wide"
)

# --- Logo de la Entidad ---
st.sidebar.image("assets/logo.png")

# --- Configuración ---
config = load_config()
ACTIVE_TOPIC = config.get("ACTIVE_TOPIC", "")
topic_config = get_active_topic_config()

# --- Interfaz Principal ---
st.title("🎛️ Panel de Control")
st.markdown(
    "Ejecuta el ciclo completo de clasificación. Los botones se activarán a medida que completes cada etapa."
)
st.markdown("#### 🚀 Espera a que carge toda la pagina, el sistema esta comprobando el estado de ejecución")

# Verificar que hay un tema activo
if not ACTIVE_TOPIC:
    st.error("❌ **No hay tema activo seleccionado**")
    st.info("👉 Ve a la página de **Configuración Inicial** para crear un proyecto.")
    if st.button("🔧 Ir a Configuración"):
        st.switch_page("pages/1_Configuracion_de_Busqueda.py")
    st.stop()

# Obtener estado del pipeline
pipeline_status = check_pipeline_status(ACTIVE_TOPIC)

st.markdown(f"### 📊 Proyecto Activo: **{ACTIVE_TOPIC.upper()}**")
st.markdown("---")

# --- SECCIÓN 1: FLUJO DE TRABAJO PRINCIPAL ---
st.markdown("### 🚀 Etapa 1: Generación de Candidatos y Validación Inicial")
st.markdown(
    "> ¡El punto de partida hacia la inteligencia! En esta etapa, el sistema explora tus datos para encontrar "
    "los primeros contratos candidatos. Tu validación inicial se convierte en la semilla fundamental que "
    "enseñará al sistema a pensar como un experto."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 1️⃣ Generar Candidatos para Revisión")
    st.markdown("Inicia el proceso analizando los datos de entrada para crear la primera lista de contratos a validar.")
    
    preprocessed_file = get_preprocessed_data_path(ACTIVE_TOPIC)
    is_step1_disabled = not preprocessed_file.exists()
    
    if is_step1_disabled:
        st.warning("⚠️ Genera el archivo de datos estandarizado primero desde la página 'Fuente de Datos'.")
        if st.button("🔧 Ir a Fuente de Datos"):
            st.switch_page("pages/0_Fuente_de_Datos.py")
    else:
        # --- Selectores de Columnas (NUEVO) ---
        try:
            df_columns = get_file_columns(str(preprocessed_file))
            
            # Selector para columnas de texto
            current_text_cols = topic_config.get("TEXT_COLUMNS_TO_COMBINE", [])
            selected_text_cols = st.multiselect(
                "Columnas de texto a analizar:",
                options=df_columns,
                default=[col for col in current_text_cols if col in df_columns],
                help="Selecciona las columnas con texto descriptivo para la búsqueda."
            )
            if selected_text_cols != current_text_cols:
                config["TOPICS"][ACTIVE_TOPIC]["TEXT_COLUMNS_TO_COMBINE"] = selected_text_cols
                save_config(config)
                st.toast("Columnas de texto actualizadas.")

            # Selector para columna UNSPSC
            unspsc_config = topic_config.get("FILTRADO_UNSPSC", {})
            current_unspsc_col = unspsc_config.get("COLUMNA_UNSPSC", None)
            if current_unspsc_col not in df_columns:
                current_unspsc_col = None

            selected_unspsc_col = st.selectbox(
                "Columna de códigos UNSPSC:",
                options=[None] + df_columns, # Permitir no seleccionar ninguna
                index=df_columns.index(current_unspsc_col) + 1 if current_unspsc_col else 0,
                format_func=lambda x: x if x else "(No usar filtro UNSPSC)",
                help="Columna para filtrar por códigos UNSPSC. Opcional."
            )
            if selected_unspsc_col != current_unspsc_col:
                config["TOPICS"][ACTIVE_TOPIC]["FILTRADO_UNSPSC"]["COLUMNA_UNSPSC"] = selected_unspsc_col
                save_config(config)
                st.toast("Columna UNSPSC actualizada.")

        except Exception as e:
            st.error(f"Error al leer columnas del Parquet: {e}")

        if st.button("🚀 Generar Candidatos", type="primary", key="workflow_step1"):
            with st.spinner("Ejecutando preprocesamiento..."):
                if ejecutar_script("1_Seleccion_Candidatos.py", show_progress_bar=True):
                    review_file = get_human_review_file_path(ACTIVE_TOPIC)
                    if review_file.exists():
                        df_review = pd.read_excel(review_file)
                        st.success(f"✅ ¡Paso 1 completado! Se generaron **{len(df_review)}** candidatos para tu revisión.")
                    else:
                        st.success("✅ ¡Paso 1 completado!")
                    st.rerun()
                else:
                    st.error("❌ Error en el Paso 1. Revisa los logs.")

with col2:
    st.markdown("#### 2️⃣ Validar Datos Manualmente")
    st.markdown(
        "Este es el único paso manual, es crucial hacerlo antes de continuar. Ve a la página de validación para revisar los candidatos generados."
    )
    
    # El "botón" aquí es una guía visual
    is_step2_disabled = pipeline_status.get("Paso 1: Generar Candidatos", {}).get("estado") != "Completado"
    
    if is_step2_disabled:
        st.warning("⚠️ Completa el Paso 1 primero para generar un archivo de revisión.")
    else:
        if st.button("📝 Ir a Validación Humana", type="primary"):
            st.switch_page("pages/7_PASO_2_Validacion_Humana.py")

    # --- INICIO DE MODIFICACIÓN: Expander siempre visible y robusto ---
    with st.expander("🔄 Opcional: Validar externamente con Excel"):
        st.info("Descarga el archivo generado, edítalo en Excel y súbelo para actualizar el sistema.")
        
        review_file_path = get_human_review_file_path(ACTIVE_TOPIC)
        
        # Botón de descarga (solo si el archivo existe)
        if review_file_path.exists():
            with open(review_file_path, "rb") as file:
                st.download_button(
                    label="⬇️ Descargar archivo de revisión",
                    data=file,
                    file_name=review_file_path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key=f"download_panel_{ACTIVE_TOPIC}"
                )
        else:
            st.button(
                "⬇️ Descargar archivo de revisión",
                disabled=True,
                help="El archivo aún no ha sido generado. Completa el Paso 1.",
                use_container_width=True
            )

        # Funcionalidad de carga (siempre disponible)
        uploaded_file = st.file_uploader(
            "📂 Sube aquí tu archivo validado",
            type=['xlsx'],
            key=f"uploader_panel_{ACTIVE_TOPIC}"
        )

        if uploaded_file is not None:
            # Guardar el archivo subido, sobrescribiendo el existente
            with open(review_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Archivo '{uploaded_file.name}' cargado y actualizado exitosamente.")
            st.rerun()
    # --- FIN DE MODIFICACIÓN ---

st.markdown("---")

# --- SECCIÓN 2: APRENDIZAJE Y CLASIFICACIÓN ---
st.markdown("### 🧠 Etapa 2: Aprendizaje y Clasificación")
st.markdown(
    "> ¡Donde la magia sucede! Aquí, el sistema absorbe tu conocimiento experto para aprender y volverse más "
    "inteligente. Culmina aplicando esta nueva capacidad para clasificar automáticamente toda tu base de datos, "
    "convirtiendo tu esfuerzo en resultados a gran escala."
)

col3, col4 = st.columns(2)

with col3:
    st.markdown("#### 3️⃣ Aprender de ti y Entrenar Modelos")
    st.markdown(
        "Ejecuta todo el ciclo de aprendizaje: analiza keywords, crea el dataset de fine-tuning y entrena el "
        "modelo experto y el clasificador."
    )
    is_step3_disabled = pipeline_status.get("Paso 2: Validación Humana", {}).get("estado") != "Listo para aprender"
    
    if is_step3_disabled:
        st.warning("⚠️ Completa algunas validaciones en el Paso 2 primero")
    else:
        if st.button("🤖 Ejecutar Ciclo de Aprendizaje", type="primary", key="workflow_step3"):
            scripts_to_run = {
                "2_Analisis_Feedback_Keywords.py": "Analizando keywords...",
                "3_Generar_Datos_FineTuning.py": "Generando dataset para fine-tuning...",
                "4_Entrenar_Modelo_Preclasificacion.py": "Entrenando modelo experto...",
                "5_Entrenamiento_Clasificador.py": "Entrenando clasificador final..."
            }
            
            overall_progress = st.progress(0, text="Iniciando ciclo de aprendizaje...")
            success = True
            total_scripts = len(scripts_to_run)

            for i, (script, description) in enumerate(scripts_to_run.items()):
                progress_percent = (i) / total_scripts
                overall_progress.progress(progress_percent, text=f"Paso {i+1}/{total_scripts}: {description}")
                
                # Determinar si el script individual debe mostrar su propia barra de progreso
                show_sub_progress = script in ["4_Entrenar_Modelo_Preclasificacion.py", "5_Entrenamiento_Clasificador.py"]
                
                if not ejecutar_script(script, show_progress_bar=show_sub_progress):
                    st.error(f"❌ El ciclo de aprendizaje falló en el script: {script}")
                    success = False
                    break
            
            if success:
                overall_progress.progress(1.0, text="¡Ciclo de aprendizaje completado!")
                st.success("🎉 ¡Todos los modelos han sido entrenados exitosamente!")
                st.rerun()
            else:
                overall_progress.empty()

with col4:
    st.markdown("#### 4️⃣ Aplicar Modelo y Generar Reporte")
    st.markdown(
        "Usa el clasificador entrenado para procesar todo el dataset y generar los archivos de resultados."
    )

    is_step4_disabled = pipeline_status.get("Paso 4: Entrenar Clasificador", {}).get("estado") != "Completado"
    
    if is_step4_disabled:
        st.warning("⚠️ Completa el entrenamiento del clasificador primero")
    else:
        # Obtener el valor guardado para el slider
        saved_threshold_float = topic_config.get("PREDICTION_THRESHOLD", 0.80)
        saved_threshold_int = int(saved_threshold_float * 100)

        # Crear el botón primero para el orden visual
        run_classification = st.button("📊 Ejecutar Clasificación", type="primary", key="workflow_step4")

        # Crear el slider después del botón
        confidence_threshold_int = st.slider(
            "Umbral para clasificar como 'SI':",
            min_value=50,
            max_value=99,
            value=saved_threshold_int,
            step=1,
            format="%d%%",
            help="Solo los contratos con una confianza mayor o igual a este valor serán clasificados como 'SI'. El valor se guarda al ejecutar la clasificación.",
            key=f"confidence_slider_panel_{ACTIVE_TOPIC}"
        )

        # Si el botón fue presionado, ejecutar la lógica
        if run_classification:
            confidence_threshold = float(confidence_threshold_int) / 100.0
            # Guardar el umbral antes de ejecutar
            config["TOPICS"][ACTIVE_TOPIC]["PREDICTION_THRESHOLD"] = confidence_threshold
            save_config(config)
            st.toast(f"Umbral de confianza guardado en {confidence_threshold:.0%}", icon="💾")

            if ejecutar_script("6_Ejecutar_Clasificador.py", show_progress_bar=True):
                st.success("✅ ¡Clasificación completada! Los resultados están listos para descargar.")
                st.rerun()
            else:
                st.error("❌ La clasificación falló. Revisa los logs.")

# --- SECCIÓN 3: Acciones Siguientes ---
st.markdown("---")
st.markdown("### 🔧 Etapa 3: Calibración y Revisión Activa")
st.markdown(
    "> ¡El camino a la maestría! En esta fase final, calibramos el sistema para alcanzar la máxima precisión. "
    "Nos enfocamos en los casos más desafiantes para el modelo, permitiendo que tu experiencia refine sus "
    "puntos débiles y garantice resultados de la más alta confianza."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📈 Análisis de Rendimiento")
    
    metrics_file = get_report_path(ACTIVE_TOPIC, 'clasificacion').with_suffix('.json')
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            accuracy = metrics.get('accuracy', 0)
            si_metrics = metrics.get('SI', {})
            si_precision = si_metrics.get('precision', 0)
            si_recall = si_metrics.get('recall', 0)

            # Mostrar métricas clave directamente
            st.metric("Precisión Global (Accuracy)", f"{accuracy:.1%}")
            st.metric(
                "Precisión para 'SI'", f"{si_precision:.1%}", help="De los que el modelo dijo 'SI', ¿cuántos acertó?"
            )
            st.metric(
                "Cobertura para 'SI' (Recall)", f"{si_recall:.1%}", help="De todos los 'SI' reales, ¿cuántos encontró el modelo?"
            )

            if st.button("Ver Historial Completo", key="view_history"):
                st.switch_page("pages/4_Historial_de_Modelos.py")
        except Exception as e:
            st.error(f"No se pudieron cargar las métricas: {e}")
    else:
        st.info("Entrena un modelo para ver sus métricas aquí.")

with col2:
    st.markdown("#### 🔄 Revisión Activa")
    
    # Obtener el valor actual desde la configuración
    active_learning_config = topic_config.get("ACTIVE_LEARNING", {})
    max_samples_current = active_learning_config.get("MAX_SAMPLES", 250)

    # Campo para que el usuario cambie el valor
    max_samples_new = st.number_input(
        "Número de muestras para revisión:",
        min_value=50,
        max_value=1000,
        value=max_samples_current,
        step=50,
        help="Define cuántos contratos (los más inciertos) se seleccionarán para la revisión activa.",
        key=f"max_samples_input_{ACTIVE_TOPIC}"
    )

    if st.button("🎯 Generar Revisión Activa", type="primary"):
        # Guardar el nuevo valor en la configuración antes de ejecutar el script
        config["TOPICS"][ACTIVE_TOPIC]["ACTIVE_LEARNING"]["MAX_SAMPLES"] = max_samples_new
        save_config(config)
        st.toast(f"Número de muestras para revisión guardado: {max_samples_new}", icon="💾")

        with st.spinner("Generando contratos para revisión activa..."):
            if ejecutar_script("7_Generar_Revision_Desde_Predicciones.py"):
                st.success("✅ Revisión activa generada exitosamente!")
                st.info("👉 Ve al Paso 2 para revisar los contratos seleccionados.")
            else:
                st.error("❌ Error al generar revisión activa.")

with col3:
    st.markdown("#### 🤝 Consolidar y Re-entrenar")
    
    # Por defecto, el botón está deshabilitado
    is_consolidate_disabled = True
    message = "⚠️ Completa algunas validaciones en el Paso 2 primero"

    # Chequeo 1: La validación principal debe haber comenzado
    if pipeline_status.get("Paso 2: Validación Humana", {}).get("estado") == "Listo para aprender":
        is_consolidate_disabled = False  # Se habilita si la validación principal está OK
        message = "" # Se limpia el mensaje por defecto

        # Chequeo 2: Si existe la revisión activa, debe estar validada en más de un 80%
        active_review_file = get_human_review_file_path(ACTIVE_TOPIC) # Usa la función correcta
        if active_review_file.exists():
            try:
                df_active = pd.read_excel(active_review_file)
                validation_col = f'Es_{ACTIVE_TOPIC.capitalize()}_Validado'
                if validation_col in df_active.columns and not df_active.empty:
                    total_rows = len(df_active)
                    validated_rows = df_active[df_active[validation_col].isin(['SI', 'NO'])].shape[0]
                    validation_percentage = (validated_rows / total_rows) * 100
                    if validation_percentage < 80:
                        is_consolidate_disabled = True  # Se deshabilita de nuevo
                        message = (
                            f"⚠️ Debes validar al menos el 80% de la revisión activa. Progreso: {validation_percentage:.1f}%"
                        )
            except Exception as e:
                is_consolidate_disabled = True
                message = f"⚠️ No se pudo leer el archivo de revisión activa: {e}"

    if is_consolidate_disabled:
        st.warning(message)
    else:
        if st.button("🚀 Iniciar Ciclo de Mejora", type="primary", key="workflow_consolidate"):
            # Scripts a ejecutar en secuencia para el ciclo completo de mejora
            scripts_to_run = {
                "8_Consolidar_Validaciones.py": "Consolidando tus validaciones...",
                "2_Analisis_Feedback_Keywords.py": "Analizando keywords...",
                "3_Generar_Datos_FineTuning.py": "Generando dataset para fine-tuning...",
                "4_Entrenar_Modelo_Preclasificacion.py": "Entrenando modelo experto...",
                "5_Entrenamiento_Clasificador.py": "Entrenando clasificador final...",
                "6_Ejecutar_Clasificador.py": "Clasificando datos masivamente..."
            }
            
            overall_progress = st.progress(0, text="Iniciando ciclo de mejora...")
            success = True
            total_scripts = len(scripts_to_run)

            for i, (script, description) in enumerate(scripts_to_run.items()):
                progress_percent = (i) / total_scripts
                overall_progress.progress(progress_percent, text=f"Paso {i+1}/{total_scripts}: {description}")
                
                # Determinar si el script individual debe mostrar su propia barra de progreso                
                show_sub_progress = script in [
                    "4_Entrenar_Modelo_Preclasificacion.py", "5_Entrenamiento_Clasificador.py", "6_Ejecutar_Clasificador.py"
                ]
                
                if not ejecutar_script(script, show_progress_bar=show_sub_progress):
                    st.error(f"❌ El ciclo de mejora falló en el script: {script}")
                    success = False
                    break
            
            if success:
                overall_progress.progress(1.0, text="¡Ciclo de mejora completado!")
                st.success(
                    "🎉 ¡Ciclo completo finalizado! Modelos re-entrenados y datos clasificados."
                )
                st.rerun()

# --- SECCIÓN 4: DESCARGA DE RESULTADOS ---
st.markdown("---")
st.markdown("### 📥 Descarga de Resultados")

predictions_csv_file = get_predictions_path(ACTIVE_TOPIC, format='csv')
predictions_excel_file = get_predictions_path(ACTIVE_TOPIC, format='xlsx')

if predictions_csv_file.exists():
    if predictions_excel_file.exists():
        st.markdown("#### ✅ Solo Contratos Relevantes (Excel)")
        st.markdown("Descarga únicamente los contratos clasificados como 'SI'")

        with open(predictions_excel_file, 'rb') as file:
            st.download_button(
                label="⬇️ Descargar Excel (Solo SI)",
                data=file.read(),
                file_name=f"predicciones_{ACTIVE_TOPIC}_SI.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
else:
    st.info("📋 Los resultados aparecerán aquí después de completar la clasificación final.")
