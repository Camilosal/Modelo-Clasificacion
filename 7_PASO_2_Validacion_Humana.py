import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from utils import (
    get_active_topic, display_validation_summary, check_pipeline_status, ejecutar_script, 
    load_config, save_config, simplificar_columna_nombre, formatear_porcentaje, 
    formatear_confianza, mostrar_ayuda_contextual, get_descripcion_amigable,
    mostrar_error_amigable, validar_archivo_con_mensaje_amigable,
    cargar_hashes_validados, anadir_hashes_validados, generar_hash_contrato,
    get_human_review_file_path, get_active_review_file_path, get_topic_history_dir
)
import io

st.set_page_config(
    page_title="Paso 2: Validación Humana",
    page_icon="assets/logo.png"
)

st.title("📝 PASO 2: Validación Humana")
st.markdown("---")

# Verificar que el tema esté configurado antes de continuar
TOPIC_NAME = get_active_topic()

# Mostrar el estado actual del paso
pipeline_status = check_pipeline_status(TOPIC_NAME)
step_status = pipeline_status.get("Paso 2: Validación Humana", {"estado": "Desconocido", "detalle": ""})
st.info(f"**Estado actual:** {step_status['estado']} - {step_status['detalle']}")
st.markdown("---")

st.markdown("""
### 🎯 ¿Qué vas a hacer aquí?

Este es el paso **más importante** de todo el proceso. Aquí revisarás contratos que el sistema encontró como posibles candidatos y decidirás si son relevantes para **{}** o no.

🔑 **Tu papel es clave**: Cada vez que marcas un contrato como "SÍ" o "NO", le estás enseñando al sistema a ser más inteligente.

📊 **Resultado**: Con tus validaciones, el sistema podrá clasificar automáticamente miles de contratos similares.
""".format(TOPIC_NAME.capitalize()))

# --- Definición de Rutas y Nombres de Columnas ---
review_file = get_human_review_file_path(TOPIC_NAME)
active_review_file = get_active_review_file_path(TOPIC_NAME)
validation_col = f'Es_{TOPIC_NAME.capitalize()}_Validado'
id_col = "ID Contrato"

# Obtener las columnas de texto configuradas en Inicio para mostrarlas
config = load_config() # Cargar configuración completa para obtener columnas
topic_config = config.get("TOPICS", {}).get(TOPIC_NAME, {})
text_columns_to_combine = topic_config.get("TEXT_COLUMNS_TO_COMBINE", ["Objeto del Contrato"])

# Definir las columnas a mostrar en el editor para una vista más limpia
display_columns = [
    id_col,
    'Metodo_Deteccion',
    validation_col,
] + text_columns_to_combine + [
    'Subtemas_Por_Keyword',
    'Similitud_Semantica_Max',
    'Keywords_Detectados'
]

# --- Pestañas para los Ciclos de Revisión ---
tab_main, tab_active = st.tabs(["Revisión Principal (1er Ciclo)", "Revisión Activa (2º Ciclo)"])

# --- Pestaña 1: Revisión Principal ---
with tab_main:
    st.header("Revisión Principal")
    st.markdown("Esta es la lista de candidatos generada en el **Paso 1**. Tu tarea es validar cada contrato como 'SI' o 'NO'.")
    if review_file.exists():
        try:
            df_review_original = pd.read_excel(review_file, engine='openpyxl')

            # --- INICIO DE LA NUEVA LÓGICA DE FILTRADO ---
            # Asegurarse de que la columna hash existe
            if 'hash_contrato' not in df_review_original.columns:
                st.warning("Generando hashes para el archivo de revisión. Esto puede tardar un momento...")
                df_review_original = generar_hash_contrato(df_review_original, text_columns_to_combine)
                df_review_original.to_excel(review_file, index=False, engine='openpyxl')

            # Cargar hashes ya validados
            hashes_validados = cargar_hashes_validados(TOPIC_NAME)
            
            # Filtrar para mostrar solo los contratos no validados previamente
            df_para_revisar = df_review_original[~df_review_original['hash_contrato'].isin(hashes_validados)].copy()
            
            num_nuevos = len(df_para_revisar)
            num_historial = len(hashes_validados)

            st.success(f"✅ Archivo de revisión cargado. Se han encontrado **{num_nuevos}** nuevos contratos para validar.")
            st.info(f"ℹ️ **{num_historial}** contratos ya han sido validados en ciclos anteriores y no se muestran aquí.")

            # --- FIN DE LA NUEVA LÓGICA DE FILTRADO ---

            if 'df_review' not in st.session_state or st.session_state.get('topic_name_for_review') != TOPIC_NAME:
                st.session_state.df_review = df_para_revisar # Usar el dataframe filtrado
                st.session_state.df_review_original = df_review_original # Guardar el original para la fusión
                st.session_state.topic_name_for_review = TOPIC_NAME

        except Exception as e:
            if not review_file.exists():
                mostrar_error_amigable("archivo_no_encontrado", 
                                     f"Archivo: {review_file.name}",
                                     "Ejecuta el Paso 1 primero para generar los contratos candidatos.")
            else:
                mostrar_error_amigable("archivo_corrupto", 
                                     f"Error técnico: {str(e)}",
                                     "Verifica que el archivo no esté abierto en Excel o intenta regenerarlo desde el Paso 1.")
            st.stop()


        # --- Controles de Interfaz (Búsqueda, Orden y Paginación) ---
        st.markdown("#### Herramientas de Revisión")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            search_term = st.text_input("🔍 Buscar en texto del contrato:", key="search_main")
        with col2:
            sort_by_uncertainty = st.button("🧬 Ordenar por Incertidumbre", key="sort_main", help="Muestra primero los contratos donde el modelo tuvo más dudas (similitud cercana a 0.5)")
        
        # Lógica de filtrado y ordenación
        df_filtered = st.session_state.df_review.copy()
        if search_term:
            # Filtrar por el término de búsqueda en las columnas de texto configuradas
            text_search_mask = df_filtered[text_columns_to_combine].apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            df_filtered = df_filtered[text_search_mask]

        if sort_by_uncertainty:
            # Calcular la incertidumbre como la distancia a 0.5 y ordenar
            df_filtered['incertidumbre'] = abs(df_filtered['Similitud_Semantica_Max'] - 0.5)
            df_filtered = df_filtered.sort_values(by='incertidumbre', ascending=True).drop(columns=['incertidumbre'])

        # Lógica de paginación
        total_rows = len(df_filtered)
        rows_per_page = st.selectbox("Contratos por página:", [25, 50, 100, 250], index=1, key=f"pagesize_main_{TOPIC_NAME}")
        total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page > 0 else 0)
        
        with col3:
            page_number = st.number_input("Página:", min_value=1, max_value=total_pages, value=1, key=f"page_main_{TOPIC_NAME}")
        
        start_idx = (page_number - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        df_paginated = df_filtered.iloc[start_idx:end_idx].copy()
        
        # Simplificar nombres de columnas para la interfaz
        df_paginated = df_paginated.rename(columns={
            'Similitud_Semantica_Max': 'Relevancia (%)',
            'Metodo_Deteccion': 'Encontrado por',
            'Keywords_Detectados': 'Palabras clave encontradas'
        })
        
        # Formatear columna de relevancia como porcentaje
        if 'Relevancia (%)' in df_paginated.columns:
            df_paginated['Relevancia (%)'] = df_paginated['Relevancia (%)'].apply(formatear_porcentaje)

        # Filtrar el dataframe para mostrar solo las columnas deseadas (actualizando nombres)
        display_columns_friendly = [col.replace('Similitud_Semantica_Max', 'Relevancia (%)').replace('Metodo_Deteccion', 'Encontrado por').replace('Keywords_Detectados', 'Palabras clave encontradas') for col in display_columns]
        existing_display_columns = [col for col in display_columns_friendly if col in df_paginated.columns]
        df_display_main = df_paginated[existing_display_columns]
        
        # Asegurar que la columna de validación sea de tipo string antes de aplicar .str methods
        df_display_main[validation_col] = df_display_main[validation_col].astype(str)

        # --- Filtro de Validación Pendiente ---
        if st.checkbox("Mostrar solo contratos pendientes de validación"):
            df_display_main = df_display_main[df_display_main[validation_col] == ""]

        # Editor Interactivo con filtros habilitados por defecto
        # Agregar ayuda contextual
        st.markdown("### 📋 Instrucciones:")
        st.info("🔍 **Relevancia (%)**: Qué tan relacionado está el contrato con el tema. Valores más altos = más relevante.")
        st.info("🎯 **Encontrado por**: Cómo el sistema encontró este contrato (palabras clave o análisis de contenido).")
        st.info("✅ **Tu tarea**: Revisa el objeto del contrato y marca SÍ si es relevante para {}, NO si no lo es.".format(TOPIC_NAME.capitalize()))
        
        edited_df_display = st.data_editor(
            df_display_main,
            num_rows="dynamic", use_container_width=True, height=600,
            column_config={
                validation_col: st.column_config.SelectboxColumn("✅ ¿Es relevante?", options=["SI", "NO", ""], required=True),
                id_col: st.column_config.TextColumn(disabled=True),
                "Relevancia (%)": st.column_config.TextColumn("🎯 Relevancia", disabled=True),
                "Encontrado por": st.column_config.TextColumn("🔍 Encontrado por", disabled=True)
            },
            key=f"editor_main_{TOPIC_NAME}"
        )

        if st.button("💾 Guardar Cambios (Revisión Principal)", key="save_main"):
            # --- INICIO DE LA NUEVA LÓGICA DE GUARDADO ---
            try:
                # 1. Identificar los contratos que realmente se validaron en esta sesión
                df_editado = edited_df_display.copy()
                df_editado[validation_col] = df_editado[validation_col].astype(str).str.strip()
                df_recien_validado = df_editado[df_editado[validation_col].isin(['SI', 'NO'])]

                if not df_recien_validado.empty:
                    # 2. Extraer los hashes de los contratos recién validados
                    hashes_a_guardar = df_recien_validado['hash_contrato'].tolist()
                    
                    # 3. Añadir los nuevos hashes al historial central
                    anadir_hashes_validados(TOPIC_NAME, hashes_a_guardar)
                    
                    # 4. Guardar una copia de esta sesión de validación con fecha y hora
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    history_path = get_topic_history_dir(TOPIC_NAME)
                    archivo_historico = history_path / f"validacion_{timestamp}.xlsx"
                    df_recien_validado.to_excel(archivo_historico, index=False, engine='openpyxl')

                    # 5. Actualizar el archivo de revisión principal (df_review_original)
                    df_original = st.session_state.df_review_original
                    df_original.set_index('hash_contrato', inplace=True)
                    df_recien_validado.set_index('hash_contrato', inplace=True)
                    df_original.update(df_recien_validado[[validation_col]])
                    df_original.reset_index(inplace=True)
                    
                    df_original.to_excel(review_file, index=False, engine='openpyxl')

                    st.success(f"¡Guardado! Se validaron {len(df_recien_validado)} contratos. El historial ha sido actualizado.")
                    st.info(f"Se ha creado un archivo de auditoría: {archivo_historico.name}")
                    
                    # Recargar la página para mostrar solo los pendientes
                    st.rerun()
                else:
                    st.warning("No se detectaron nuevas validaciones ('SI' o 'NO'). No se guardó nada.")

            except Exception as e:
                mostrar_error_amigable("archivo_corrupto", f"Error al guardar: {e}", "Inténtalo de nuevo.")
            # --- FIN DE LA NUEVA LÓGICA DE GUARDADO ---

        # --- Carga y Descarga ---
        st.markdown("---")
        st.markdown("#### Carga y Descarga (para edición externa)")
        col1, col2 = st.columns(2)
        with col1:
            # Descargar
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                st.session_state.df_review.to_excel(writer, index=False, sheet_name='Revision')
            st.download_button("📥 Descargar para editar en Excel", data=output.getvalue(), file_name=review_file.name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_main")
        with col2:
            # Cargar
            uploaded_file = st.file_uploader("📂 Sube aquí tu archivo validado (.xlsx)", type=['xlsx'], key="uploader_main")
            if uploaded_file is not None:
                try:
                    df_uploaded = pd.read_excel(uploaded_file, engine='openpyxl')
                    df_uploaded.to_excel(review_file, index=False, engine='openpyxl')
                    st.session_state.df_review = df_uploaded
                    st.success(f"Archivo '{uploaded_file.name}' cargado y guardado. La tabla se ha actualizado.")
                    st.rerun()
                except Exception as e:
                    mostrar_error_amigable("archivo_corrupto", 
                                         f"Archivo: {uploaded_file.name}",
                                         "Verifica que el archivo tenga el formato correcto y no esté dañado.")

# --- Pestaña 2: Revisión Activa ---
with tab_active:
    st.header("Revisión Activa (Segundo Ciclo)")
    st.markdown("""
    Esta lista contiene los casos más valiosos para el aprendizaje del modelo, generados a partir de la clasificación masiva.
    
    Tu tarea es **confirmar o corregir** la predicción del modelo para cada contrato.
    - Si el modelo acertó, selecciona la misma opción ('SI' o 'NO') en la columna "Tu Validación".
    - Si el modelo se equivocó, selecciona la respuesta correcta.
    """)

    if not active_review_file.exists():
        st.info("Aún no has generado una lista de revisión activa. Puedes hacerlo desde el 'Paso 5' después de ejecutar una clasificación masiva.")
    else:
        try:
            # Cargar el dataframe de revisión activa si no está en el estado de la sesión
            if 'df_active_review' not in st.session_state or st.session_state.get('topic_name_for_active_review') != TOPIC_NAME:
                st.session_state.df_active_review = pd.read_excel(active_review_file, engine='openpyxl')
                st.session_state.topic_name_for_active_review = TOPIC_NAME

            # Lógica de la interfaz de usuario (reutilizando la del primer ciclo)
            ra_search_term = st.text_input("🔍 Buscar en texto del contrato:", key="search_active")
            ra_sort_by_uncertainty = st.button("🧬 Ordenar por Incertidumbre", key="sort_active", help="Muestra primero los contratos donde el modelo tuvo más dudas (confianza cercana a 0.5)")

            ra_df_filtered = st.session_state.df_active_review.copy()
            if ra_search_term:
                ra_text_search_mask = ra_df_filtered[text_columns_to_combine].apply(lambda x: x.str.contains(ra_search_term, case=False, na=False)).any(axis=1)
                ra_df_filtered = ra_df_filtered[ra_text_search_mask]

            if ra_sort_by_uncertainty:
                ra_df_filtered['incertidumbre'] = abs(ra_df_filtered[f'Confianza_{TOPIC_NAME.capitalize()}_SI'] - 0.5)
                ra_df_filtered = ra_df_filtered.sort_values(by='incertidumbre', ascending=True).drop(columns=['incertidumbre'])

            ra_total_rows = len(ra_df_filtered)
            ra_rows_per_page = st.selectbox("Contratos por página:", [25, 50, 100, 250], index=1, key=f"pagesize_active_{TOPIC_NAME}")
            ra_total_pages = (ra_total_rows // ra_rows_per_page) + (1 if ra_total_rows % ra_rows_per_page > 0 else 0)
            ra_page_number = st.number_input("Página:", min_value=1, max_value=ra_total_pages, value=1, key=f"page_active_{TOPIC_NAME}")

            ra_start_idx = (ra_page_number - 1) * ra_rows_per_page
            ra_end_idx = ra_start_idx + ra_rows_per_page
            ra_df_paginated = ra_df_filtered.iloc[ra_start_idx:ra_end_idx]
            
            # --- [MEJORA] Definir columnas específicas y relevantes para la revisión activa ---
            prediction_col = f'Prediccion_{TOPIC_NAME.capitalize()}'
            confidence_col = f'Confianza_{TOPIC_NAME.capitalize()}_SI'
            
            active_review_display_columns = [
                id_col,
                prediction_col,
                validation_col,
            ] + text_columns_to_combine + [
                confidence_col,
                'Subtema_Detectado'
            ]

            # Filtrar para mostrar solo las columnas relevantes que existen en el dataframe
            existing_display_columns_active = [col for col in active_review_display_columns if col in ra_df_paginated.columns]
            df_display_active = ra_df_paginated[existing_display_columns_active].copy()

            if st.checkbox("Mostrar solo contratos pendientes de validación (Revisión Activa)", key="filter_active"):
                # Asegurarse de que la columna de validación exista y manejar valores nulos
                if validation_col in df_display_active.columns:
                    df_display_active = df_display_active[~df_display_active[validation_col].isin(['SI', 'NO'])]

            edited_df_active_display = st.data_editor(
                df_display_active,
                num_rows="dynamic", use_container_width=True, height=600,
                column_config={
                    validation_col: st.column_config.SelectboxColumn("✅ Tu Validación", options=["SI", "NO", ""], required=True, help="Corrige la predicción del modelo si es necesario."),
                    id_col: st.column_config.TextColumn("ID Contrato", disabled=True),
                    prediction_col: st.column_config.TextColumn("🤖 Predicción Modelo", disabled=True),
                    confidence_col: st.column_config.ProgressColumn(
                        "🎯 Confianza Modelo",
                        help="Nivel de confianza del modelo en su predicción 'SI'",
                        format="%.2f",
                        min_value=0.0,
                        max_value=1.0,
                    ),
                    'Subtema_Detectado': st.column_config.TextColumn("Subtema Detectado", disabled=True)
                },
                key=f"editor_active_{TOPIC_NAME}"
            )

            if st.button("💾 Guardar Cambios (Revisión Activa)", key="save_active"):
                st.session_state.df_active_review.set_index(id_col, inplace=True)
                edited_df_active_display.set_index(id_col, inplace=True)
                st.session_state.df_active_review.update(edited_df_active_display)
                st.session_state.df_active_review.reset_index(inplace=True)

                st.session_state.df_active_review.to_excel(active_review_file, index=False, engine='openpyxl')
                st.success("¡Cambios guardados en la revisión activa!")

        except Exception as e:
            mostrar_error_amigable("archivo_corrupto", f"Error técnico: {str(e)}", "Verifica que el archivo no esté abierto en Excel o intenta regenerarlo desde el Paso 7.")
            st.stop()

        # --- Carga y Descarga (Revisión Activa) ---
        st.markdown("---")
        st.markdown("#### Carga y Descarga (para edición externa)")
        col1_active, col2_active = st.columns(2)
        with col1_active:
            output_active = io.BytesIO()
            with pd.ExcelWriter(output_active, engine='openpyxl') as writer:
                st.session_state.df_active_review.to_excel(writer, index=False, sheet_name='RevisionActiva')
            st.download_button("📥 Descargar para editar en Excel", data=output_active.getvalue(), file_name=active_review_file.name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_active")
        with col2_active:
            uploaded_file_active = st.file_uploader("📂 Sube aquí tu archivo validado (.xlsx)", type=['xlsx'], key="uploader_active")
            if uploaded_file_active is not None:
                try:
                    df_uploaded_active = pd.read_excel(uploaded_file_active, engine='openpyxl')
                    df_uploaded_active.to_excel(active_review_file, index=False, engine='openpyxl')
                    st.session_state.df_active_review = df_uploaded_active
                    st.success(f"Archivo '{uploaded_file_active.name}' cargado y guardado. La tabla se ha actualizado.")
                    st.rerun()
                except Exception as e:
                    mostrar_error_amigable("archivo_corrupto", 
                                         f"Archivo: {uploaded_file.name}",
                                         "Verifica que el archivo tenga el formato correcto y no esté dañado.")

        st.markdown("---")
        st.markdown("### Consolidar Todas las Validaciones")
        st.warning("**Paso Final del Ciclo de Retroalimentación:** Esta acción combinará las validaciones de la 'Revisión Principal' y la 'Revisión Activa' en un único archivo de entrenamiento. Las validaciones de la revisión activa tendrán prioridad si un contrato aparece en ambas listas.")
        
        if st.button("🤝 Consolidar Todas las Validaciones"):
            if ejecutar_script("8_Consolidar_Validaciones.py"):
                # Marcar en la configuración que el ciclo se ha completado
                config = load_config()
                topic_config = config.get("TOPICS", {}).get(TOPIC_NAME, {})
                topic_config["active_loop_completed"] = True
                save_config(config)

                st.success("¡Validaciones consolidadas! El dataset de entrenamiento ha sido enriquecido.")
                st.info("Ahora puedes ir al **Paso 4: Entrenar Clasificador** para re-entrenar un modelo mucho más robusto.")
                # Borrar el estado para forzar la recarga del archivo principal la próxima vez
                if 'df_review' in st.session_state:
                    del st.session_state['df_review']
                st.rerun()
