import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# --- Importar funciones de utilidad ---
from utils import (
    load_keywords, save_keywords, load_exclusion_words, save_exclusion_words,
    load_config, save_config, check_pipeline_status, borrar_resultados_por_tema, get_active_topic,
    INPUT_DIR, RESULTS_DIR, BASE_DIR, CONFIG_FILE, get_file_columns, get_topic_input_dir, get_topic_results_dir,
    get_keywords_file_path, get_exclusion_file_path, get_report_path, get_preprocessed_data_path,
    mostrar_ayuda_contextual, crear_tooltip, mostrar_error_amigable, validar_archivo_con_mensaje_amigable, display_file_explorer,
    crear_dashboard_estado, mostrar_dashboard_visual, mostrar_boton_wizard, mostrar_wizard_configuracion,
    ejecutar_script
)

st.set_page_config(
    page_title="Configuración Inicial - Sistema de Clasificación",
    page_icon="⚙️",
    layout="wide"
)

# --- Logo de la Entidad ---
st.sidebar.image("assets/logo.png")

# --- Sección "Acerca de" en la barra lateral ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Acerca del Sistema")
st.sidebar.info(
    """
    **Versión:** 1.0.1

    **Desarrollado por:** Manuel Camilo Saldarriaga - GAEC/EMAE

    **Contacto:** camilosal@me.com

    """
)

# --- Cargar configuración ---
config = load_config()
ACTIVE_TOPIC = config.get("ACTIVE_TOPIC", "")

# Cargar la configuración específica del tema y definir las variables del archivo de entrada
topic_config = config.get("TOPICS", {}).get(ACTIVE_TOPIC, {})

# [CORRECCIÓN] Obtener la ruta del archivo de datos desde la nueva estructura DATA_SOURCE_CONFIG
input_file_name = ""
input_data_file = None
if topic_config:
    # Se prioriza la ruta del CSV, ya que es la fuente más común y necesaria para leer columnas.
    source_config = topic_config.get("DATA_SOURCE_CONFIG", {}).get("CSV", {})
    file_name = source_config.get("FILENAME")
    if file_name:
        input_data_file = get_topic_input_dir(ACTIVE_TOPIC) / file_name
        input_file_name = file_name

# --- Interfaz Principal ---
st.title("⚙️ Configuración Inicial")
st.markdown("Configura los parámetros básicos para tu proyecto de clasificación de contratos.")

# --- Wizard de Configuración ---
st.markdown("---")
st.markdown("### 🧙‍♂️ Configuración Rápida")
st.markdown("¿Primera vez usando el sistema? Usa el asistente para configurar todo automáticamente.")

# Mostrar wizard siempre disponible
col1, col2 = st.columns(2)

with col1:
    st.markdown("**🚀 Asistente Guiado**")
    st.markdown("Configura paso a paso con ayuda")
    
    if st.button("🧙‍♂️ Iniciar Asistente", type="primary", key="wizard_config"):
        st.session_state.mostrar_wizard = True
        st.rerun()

with col2:
    st.markdown("**⚙️ Configuración Manual**")
    st.markdown("Usa las opciones avanzadas más abajo")
    
    if st.button("⬇️ Ir a Configuración Manual", key="manual_config"):
        # Scroll hacia abajo a la configuración manual
        st.markdown("👇 **Configuración manual disponible más abajo**")

# Mostrar wizard si está activo
if st.session_state.get('mostrar_wizard', False):
    st.markdown("---")
    mostrar_wizard_configuracion()
    
    if st.button("❌ Cerrar Asistente", type="secondary", key="close_wizard"):
        st.session_state.mostrar_wizard = False
        st.rerun()

st.markdown("---")

# --- Gestor de Temas de Análisis ---
st.markdown("### 🎯 Paso A: Gestionar Temas de Análisis")
st.markdown("""
**¿Qué es un tema?** Es el área específica en la que quieres clasificar contratos (ej: 'ciberseguridad', 'obras civiles', 'consultoría').

Cada tema tiene:
- 📝 Sus propias palabras clave
- 🤖 Su propio modelo de aprendizaje  
- 📊 Sus propios resultados y métricas

💡 **Consejo:** Mantén temas específicos para mejores resultados (ej: 'ciberseguridad' en lugar de 'tecnología').
""")

# Agregar ayuda contextual
st.info("ℹ️ **Ayuda:** Un tema representa un área específica de clasificación. Puedes trabajar en múltiples temas sin que se interfieran entre sí.")

all_topics = list(config.get("TOPICS", {}).keys())

try:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Tema Activo")
        if all_topics:
            selected_topic = st.selectbox(
                "Selecciona el tema en el que quieres trabajar:",
                options=all_topics,
                index=all_topics.index(ACTIVE_TOPIC) if ACTIVE_TOPIC in all_topics else 0
            )

            # Lógica para cambiar de tema
            if selected_topic and selected_topic != ACTIVE_TOPIC:
                config["ACTIVE_TOPIC"] = selected_topic
                save_config(config)
                st.success(f"✅ Tema activo cambiado a: **{selected_topic}**")
                st.rerun()

            # Mostrar y permitir la edición de la descripción y contexto del tema activo
            if ACTIVE_TOPIC:
                active_topic_details = config.get("TOPICS", {}).get(ACTIVE_TOPIC, {})
                st.info(f"**Descripción:** {active_topic_details.get('descripcion', 'No definida')}")
                st.caption(f"**Contexto:** {active_topic_details.get('contexto', 'No definido')}")

                with st.expander("✏️ Editar Descripción y Contexto del Tema Activo"):
                    new_desc = st.text_area(
                        "Descripción:",
                        value=active_topic_details.get('descripcion', ''),
                        key=f"desc_edit_{ACTIVE_TOPIC}" # Clave dinámica para evitar conflictos
                    )
                    new_context = st.text_input(
                        "Contexto:",
                        value=active_topic_details.get('contexto', ''),
                        key=f"context_edit_{ACTIVE_TOPIC}"
                    )

                    if st.button("Guardar Cambios de Descripción"):
                        config["TOPICS"][ACTIVE_TOPIC]['descripcion'] = new_desc
                        config["TOPICS"][ACTIVE_TOPIC]['contexto'] = new_context
                        save_config(config)
                        st.success("✅ Descripción y contexto actualizados.")
                        st.rerun()
        else:
            st.info("📝 No hay temas creados aún. Crea uno nuevo a continuación.")
    
    with col2:
        st.markdown("#### ➕ Crear Nuevo Tema")
        with st.form("new_topic_form"):
            new_topic_name = st.text_input(
                "Nombre del nuevo tema:",
                placeholder="Ej: ciberseguridad, obras_civiles, consultoria"
            )
            new_topic_desc = st.text_area(
                "Descripción del Tema:",
                placeholder="Ej: Clasificar contratos relacionados con la adquisición de software y servicios de ciberseguridad.",
                help="Una breve explicación del objetivo de la clasificación para este tema."
            )
            new_topic_context = st.text_input(
                "Contexto (Sector/Organización):",
                placeholder="Ej: Sector Financiero, Departamento de TI",
                help="El sector, organización o departamento al que se aplica este tema."
            )

            if st.form_submit_button("Crear Tema"):
                if new_topic_name and new_topic_name.strip() and new_topic_desc and new_topic_desc.strip():
                    clean_topic_name = new_topic_name.strip().lower().replace(" ", "_")
                    
                    if clean_topic_name not in config.get("TOPICS", {}):
                        # Crear una estructura de configuración completa y por defecto para el nuevo tema
                        config["TOPICS"][clean_topic_name] = {
                            "descripcion": new_topic_desc.strip(),
                            "contexto": new_topic_context.strip(),
                            "DATA_SOURCE_CONFIG": {
                                "ACTIVE_SOURCE": "CSV",
                                "API": {"BASE_URL": "", "API_KEY": "", "QUERY": ""},
                                "CSV": {"FILENAME": ""},
                                "SQL": {"DB_TYPE": "postgresql", "HOST": "", "PORT": "", "DATABASE": "", "USERNAME": "", "PASSWORD": "", "DRIVER": "ODBC Driver 17 for SQL Server", "QUERY": ""}
                            },
                            "TEXT_COLUMNS_TO_COMBINE": ["Objeto del Contrato"],
                            "FILTRADO_UNSPSC": {
                                "descripcion": "Códigos UNSPSC para este tema",
                                "CODIGOS_DE_INTERES": []
                            },
                            "CLASSIFIER_MODEL": "RandomForestClassifier",
                            "SIMILARITY_THRESHOLD": 0.7,
                            "PREDICTION_THRESHOLD": 0.85,
                            "FINETUNING": {
                                "NUM_EPOCHS": 2,
                                "BATCH_SIZE": 16,
                                "LEARNING_RATE": 3e-05,
                                "WARMUP_RATIO": 0.1
                            },
                            "ACTIVE_LEARNING": {
                                "UNCERTAINTY_THRESHOLD_LOW": 0.7,
                                "UNCERTAINTY_THRESHOLD_HIGH": 0.9,
                                "MAX_SAMPLES": 250
                            }
                        }
                        config["ACTIVE_TOPIC"] = clean_topic_name
                        save_config(config)
                        st.success(f"✅ Tema **{clean_topic_name}** creado y activado!")
                        st.rerun()
                    else:
                        st.error("❌ Este tema ya existe. Usa un nombre diferente.")
                else:
                    st.error("❌ Debes ingresar al menos un nombre y una descripción para el tema.")

except Exception as e:
    st.error(f"❌ Error al gestionar temas: {str(e)}")

# --- Eliminar Temas ---
if all_topics:
    st.markdown("#### 🗑️ Eliminar Tema")
    with st.expander("⚠️ Eliminar tema existente"):
        st.warning("**¡Cuidado!** Esta acción eliminará permanentemente el tema y todos sus datos asociados.")
        
        topic_to_delete = st.selectbox(
            "Selecciona el tema a eliminar:",
            options=all_topics,
            key="delete_topic_select"
        )
        
        if st.button("🗑️ Eliminar tema '{topic_to_delete}' permanentemente"):
            # Borrar archivos de resultados y la carpeta del tema
            borrar_resultados_por_tema(topic_to_delete)
            # Borrar archivos de configuración (keywords, exclusion)
            get_keywords_file_path(topic_to_delete).unlink(missing_ok=True)
            get_exclusion_file_path(topic_to_delete).unlink(missing_ok=True)
            # Eliminar del config.json
            del config["TOPICS"][topic_to_delete]
            # Si el tema eliminado era el activo, deseleccionarlo
            if config["ACTIVE_TOPIC"] == topic_to_delete:
                config["ACTIVE_TOPIC"] = ""
            save_config(config)
            st.success(f"✅ Tema **{topic_to_delete}** eliminado exitosamente.")
            st.rerun()

    st.markdown("#### 💾 Copia de Seguridad del Tema Activo")
    st.info(f"Crea un respaldo completo del tema '{ACTIVE_TOPIC}', incluyendo configuración, modelos y datos de validación.")
    if st.button(f"📦 Crear Copia de Seguridad para '{ACTIVE_TOPIC}'"):
        with st.spinner(f"Creando copia de seguridad para el tema '{ACTIVE_TOPIC}'..."):
            if ejecutar_script("create_backup.py", args=[ACTIVE_TOPIC]):
                st.success("¡Copia de seguridad creada exitosamente en la carpeta 'backups'!")
            else:
                st.error("Ocurrió un error durante la copia de seguridad. Revisa los logs.")

st.markdown("---")

# --- Personalización del Proyecto ---
if ACTIVE_TOPIC:
    st.markdown("### 🎨 Paso C: Personaliza tu Proyecto")
    st.markdown(f"**Configura los detalles específicos para el tema: `{ACTIVE_TOPIC}`**")
    
    # --- Gestión de Keywords ---
    keywords_file = get_keywords_file_path(ACTIVE_TOPIC)
    with st.expander("📝 Gestionar Palabras Clave (Keywords)", expanded=True):
        st.markdown("""
        ### 🔑 ¿Qué son las palabras clave?
        Son los términos y frases que el sistema usará para la **búsqueda inicial** de contratos. Son el punto de partida para encontrar documentos relevantes.
        - **Consejo:** Usa términos específicos, sinónimos y variaciones. Agrúpalos por sub-temas para una mejor organización.
        """)
        
        # Cargar keywords en el estado de la sesión para edición
        if 'keywords' not in st.session_state or st.session_state.get('topic_name_for_keywords') != ACTIVE_TOPIC:
            st.session_state.keywords = load_keywords(keywords_file)
            st.session_state.topic_name_for_keywords = ACTIVE_TOPIC

        # Cargar datos de precisión si existen para mostrar en la UI
        report_file = get_report_path(ACTIVE_TOPIC, "rendimiento_keywords")
        df_precision = pd.DataFrame()
        if report_file.exists():
            try:
                df_precision = pd.read_excel(report_file, sheet_name='Rendimiento por Keyword')
            except Exception:
                pass # No es crítico si falla

        # UI para seleccionar el sub-tema de keywords
        st.markdown("#### 1. Selecciona un Sub-Tema para Editar")
        existing_themes = list(st.session_state.keywords.keys())
        
        # Si no hay temas, se empieza con 'general'
        if not existing_themes:
            st.session_state.keywords['general'] = []
            existing_themes = ['general']

        theme_to_edit = st.selectbox(
            "Sub-tema de keywords:",
            options=existing_themes,
            index=0,
            key=f"theme_selector_{ACTIVE_TOPIC}"
        )

        # UI para editar las keywords del sub-tema seleccionado
        st.markdown(f"#### 2. Edita las Palabras Clave para '{theme_to_edit}'")
        keywords_for_theme = st.session_state.keywords.get(theme_to_edit, [])
        
        keywords_text = st.text_area(
            "Escribe una palabra clave por línea:",
            value='\n'.join(keywords_for_theme),
            height=250,
            key=f"keywords_textarea_{ACTIVE_TOPIC}_{theme_to_edit}" # Key dinámica para refrescar
        )

        if st.button(f"💾 Guardar Cambios para '{theme_to_edit}'", type="primary"):
            updated_keywords = [k.strip() for k in keywords_text.split('\n') if k.strip()]
            st.session_state.keywords[theme_to_edit] = updated_keywords
            save_keywords(keywords_file, st.session_state.keywords)
            st.success(f"✅ Palabras clave para '{theme_to_edit}' guardadas exitosamente.")
            st.rerun()

        st.markdown("---")
        
        # UI para añadir un nuevo sub-tema
        st.markdown("#### 3. Añadir un Nuevo Sub-Tema (Opcional)")
        new_theme_name = st.text_input("Nombre del nuevo sub-tema:", placeholder="Ej: software, hardware, consultoria_ti")
        if st.button("➕ Crear Nuevo Sub-Tema"):
            if new_theme_name and new_theme_name.strip():
                clean_name = new_theme_name.strip().lower().replace(" ", "_")
                if clean_name not in st.session_state.keywords:
                    st.session_state.keywords[clean_name] = []
                    save_keywords(keywords_file, st.session_state.keywords)
                    st.success(f"✅ Sub-tema '{clean_name}' creado. Ahora puedes seleccionarlo arriba para añadirle keywords.")
                    st.rerun()
                else:
                    st.warning(f"⚠️ El sub-tema '{clean_name}' ya existe.")
            else:
                st.error("❌ Por favor, introduce un nombre para el nuevo sub-tema.")

        st.markdown("---")
        
        # UI para eliminar un sub-tema
        st.markdown("#### 4. Eliminar un Sub-Tema Existente")
        st.warning("⚠️ **¡Atención!** Esta acción es irreversible. Se eliminará el sub-tema y todas las palabras clave que contiene.")
        
        # Opciones para eliminar, excluyendo el último tema
        themes_available_for_deletion = [theme for theme in existing_themes if len(existing_themes) > 1]
        
        if themes_available_for_deletion:
            theme_to_delete = st.selectbox(
                "Selecciona el sub-tema a eliminar:",
                options=themes_available_for_deletion,
                key=f"delete_theme_selector_{ACTIVE_TOPIC}",
                index=None,
                placeholder="Elige un sub-tema para borrar..."
            )

            if st.button("🗑️ Eliminar Sub-Tema Seleccionado", disabled=not theme_to_delete):
                if theme_to_delete and theme_to_delete in st.session_state.keywords:
                    del st.session_state.keywords[theme_to_delete]
                    save_keywords(keywords_file, st.session_state.keywords)
                    st.success(f"✅ Sub-tema '{theme_to_delete}' y sus keywords han sido eliminados.")
                    st.rerun()
        else:
            st.info("No puedes eliminar el único sub-tema existente.")


        # UI para mostrar la precisión de las keywords del tema seleccionado
        if not df_precision.empty and theme_to_edit:
            st.markdown(f"#### Rendimiento de Keywords para '{theme_to_edit}'")
            keywords_in_theme = st.session_state.keywords.get(theme_to_edit, [])
            df_precision_filtered = df_precision[df_precision['Keyword'].isin(keywords_in_theme)]
            if not df_precision_filtered.empty:
                st.dataframe(
                    df_precision_filtered[['Keyword', 'Precision', 'Aciertos', 'Fallos']].style.format({'Precision': '{:.1%}'}),
                    use_container_width=True
                )
            else:
                st.info("Aún no hay datos de rendimiento para las keywords de este sub-tema. Genera un reporte en el Paso 3.")
    
    # --- Gestión de Exclusiones ---
    exclusion_file = get_exclusion_file_path(ACTIVE_TOPIC)
    with st.expander("🚫 Gestionar Palabras de Exclusión"):
        st.markdown("""
        ### ❌ ¿Qué son las palabras de exclusión?
        Son términos que **NO** quieres que aparezcan en los resultados, incluso si contienen keywords relevantes.
        
        **Ejemplos útiles:**
        - Para ciberseguridad: "seguridad industrial", "seguridad física"
        - Para obras civiles: "software", "consultoría"
        """)
        
        # Cargar exclusiones
        if 'exclusions' not in st.session_state or st.session_state.get('topic_name_for_exclusions') != ACTIVE_TOPIC:
            st.session_state.exclusions = load_exclusion_words(exclusion_file)
            st.session_state.topic_name_for_exclusions = ACTIVE_TOPIC
        
        # Editor de exclusiones
        exclusions_text = st.text_area(
            "Palabras de exclusión (una por línea):",
            value='\n'.join(st.session_state.exclusions),
            height=150,
            help="Estas palabras excluirán contratos de los resultados"
        )
        
        if st.button("💾 Guardar Exclusiones"):
            new_exclusions = [e.strip() for e in exclusions_text.split('\n') if e.strip()]
            save_exclusion_words(exclusion_file, new_exclusions)
            st.session_state.exclusions = new_exclusions
            st.success("✅ Palabras de exclusión guardadas!")
            st.rerun()
    
    # --- Configuración de Códigos UNSPSC ---
    with st.expander("🏷️ Configurar Códigos UNSPSC"):
        st.markdown("""
        ### 📋 ¿Qué son los códigos UNSPSC?
        Son códigos estándar que clasifican productos y servicios. El sistema los usa para filtrar contratos por categorías específicas en la fase de generacion de candidatos.
        
        **Ejemplo:** Para ciberseguridad, podrías usar códigos como 4323 (Software de seguridad) o 8111 (Servicios de seguridad informática).
        """)
        current_codes = config.get("TOPICS", {}).get(ACTIVE_TOPIC, {}).get("FILTRADO_UNSPSC", {}).get("CODIGOS_DE_INTERES", [])
        
        codes_text = st.text_area(
            "Códigos UNSPSC (uno por línea):",
            value='\n'.join(map(str, current_codes)),
            height=100,
            help="Estos códigos se usarán para filtrar contratos por categoría"
        )
        
        if st.button("💾 Guardar Códigos UNSPSC"):
            try:
                new_codes = [int(c.strip()) for c in codes_text.split('\n') if c.strip()]
                config["TOPICS"][ACTIVE_TOPIC]["FILTRADO_UNSPSC"]["CODIGOS_DE_INTERES"] = new_codes
                save_config(config)
                st.success("✅ Códigos UNSPSC guardados!")
                st.rerun()
            except ValueError:
                st.error("❌ Usa solo números, uno por línea.")
    
    # --- Explorador de Archivos ---
    with st.expander("📂 Explorador de Archivos del Tema"):
        st.info(f"Explora y descarga los archivos de entrada y salida generados para el tema **'{ACTIVE_TOPIC}'**.")
        
        # Obtener directorios del tema
        input_dir = get_topic_input_dir(ACTIVE_TOPIC)
        results_dir = get_topic_results_dir(ACTIVE_TOPIC)
        
        # Mostrar explorador para archivos de entrada
        display_file_explorer(input_dir, "📥 Archivos de Entrada")
        
        st.markdown("---")
        
        # Mostrar explorador para archivos de salida
        display_file_explorer(results_dir, "📤 Archivos de Salida (Resultados)")
            
# --- Mensaje estado configuracion ---
if ACTIVE_TOPIC:
    st.success("✅ **Configuración básica completada!**")
        
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎛️ Ir al Panel de Control"):
            st.switch_page("pages/2_Panel_de_Control.py")
    
    with col2:
        if st.button("🏠 Volver a Estado del Proyecto"):
            st.switch_page("pages/3_Estado_del_Proyecto.py")
else:
    st.info("📋 **Completa la configuración básica:**")
    steps = []
    if not ACTIVE_TOPIC:
        steps.append("1. Crea o selecciona un tema activo")
    if ACTIVE_TOPIC and not input_data_file:
        steps.append("2. Configura un archivo de datos de entrada")
    
    for step in steps:
        st.write(f"- {step}")

st.markdown("---")