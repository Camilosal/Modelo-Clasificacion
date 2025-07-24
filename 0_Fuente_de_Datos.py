import streamlit as st
import pandas as pd
import time
from datetime import datetime

# --- Importar funciones de utilidad ---
from utils import (
    load_config, save_config, get_active_topic,
    obtener_columnas_fuente_datos, cargar_datos_y_aplicar_filtros,
    get_topic_input_dir
)

st.set_page_config(
    page_title="Fuente de Datos - Sistema de Clasificación",
    page_icon="🗃️",
    layout="wide"
)

# --- Logo de la Entidad ---
st.sidebar.image("assets/logo.png", use_container_width=True)

# --- Cargar configuración ---
if 'config' not in st.session_state:
    st.session_state.config = load_config()

config = st.session_state.config
ACTIVE_TOPIC = get_active_topic()

# --- Interfaz Principal ---
st.title("🗃️ Configuración de Fuente de Datos y Prefiltrado")

# --- Verificación de Tema Activo ---
if not ACTIVE_TOPIC:
    st.error("❌ **No hay tema activo seleccionado**")
    st.info("👉 Ve a la página de **Configuración de Búsqueda** para crear o seleccionar un proyecto.")
    if st.button("🔧 Ir a Configuración"):
        st.switch_page("pages/1_Configuracion_de_Busqueda.py")
    st.stop()

st.markdown(f"Estás configurando la fuente de datos para el tema: **{ACTIVE_TOPIC.upper()}**")
st.markdown("Define de dónde se obtendrán los datos y aplica filtros iniciales para optimizar el análisis.")
st.markdown("---")

# [FIX] Acceder a la configuración del tema activo para todas las operaciones
topic_config = config.get("TOPICS", {}).get(ACTIVE_TOPIC, {})
if not topic_config:
    st.error(f"No se encontró la configuración para el tema '{ACTIVE_TOPIC}'. Por favor, créala o selecciónala de nuevo.")
    st.stop()

# --- 1. Selección de Fuente de Datos ---
st.markdown("### 1. Origen de los Datos")

data_source_config = topic_config.get("DATA_SOURCE_CONFIG", {})
active_source = data_source_config.get("ACTIVE_SOURCE", "CSV")

# Mapeo para una UI más amigable
source_options_map = {"(CSV/Excel)": "CSV", "API Genérica": "API", "API Datos Abiertos": "API_SODA", "Bodega de Datos": "SQL"}
source_options_display = list(source_options_map.keys())

# Encontrar el nombre amigable de la fuente activa
active_source_display_list = [k for k, v in source_options_map.items() if v == active_source]
active_source_display = active_source_display_list[0] if active_source_display_list else source_options_display[0]

selected_source_display = st.radio(
    "Selecciona la fuente de datos principal:",
    options=source_options_display,
    index=source_options_display.index(active_source_display),
    horizontal=True,
    help="Elige si cargarás los datos desde un archivo (CSV/Excel), una API o una base de datos SQL."
)

selected_source = source_options_map[selected_source_display]

if selected_source != active_source:
    config["TOPICS"][ACTIVE_TOPIC]["DATA_SOURCE_CONFIG"]["ACTIVE_SOURCE"] = selected_source
    save_config(config)
    st.session_state.config = config # Actualizar estado
    st.success(f"Fuente de datos cambiada a **{selected_source}**.")
    time.sleep(1)
    st.rerun()

# --- 2. Configuración Específica de la Fuente ---
if selected_source == "API_SODA":
    st.markdown("#### Configuración de API de Datos Abiertos (Socrata)")
    st.info("Asegúrate de haber configurado tus credenciales (`DATOS_GOV_USER`, `DATOS_GOV_PASS`, `DATOS_GOV_TOKEN`) en el archivo `.env`.")
    
    soda_config = data_source_config.get("API_SODA", {})
    
    soda_domain = st.text_input("Dominio de la API (ej: www.datos.gov.co)", value=soda_config.get("DOMAIN", "www.datos.gov.co"))
    soda_dataset_id = st.text_input("Identificador del Conjunto de Datos", value=soda_config.get("DATASET_ID", ""), placeholder="p6dx-8zbt")
    soda_select = st.text_area("Cláusula SELECT", value=soda_config.get("SELECT_CLAUSE", ""), height=150, help="Define las columnas que quieres obtener. Ejemplo: `columna1`, `columna2`")
    soda_where = st.text_area("Cláusula WHERE", value=soda_config.get("WHERE_CLAUSE", ""), height=150, help="Define los filtros para los datos. Ejemplo: `columna1` = 'valor' AND `columna2` > 100")

    if st.button("Guardar Configuración de API Datos Abiertos"):
        config["TOPICS"][ACTIVE_TOPIC]["DATA_SOURCE_CONFIG"]["API_SODA"] = {
            "DOMAIN": soda_domain, 
            "DATASET_ID": soda_dataset_id, 
            "SELECT_CLAUSE": soda_select,
            "WHERE_CLAUSE": soda_where
        }
        save_config(config)
        st.session_state.config = config
        st.success("✅ Configuración de la API de Datos Abiertos guardada.")

elif selected_source == "API":
    st.markdown("#### Configuración de la API Genérica")
    api_config = data_source_config.get("API", {})
    
    api_url = st.text_input("URL Base de la API", value=api_config.get("BASE_URL", ""), placeholder="https://api.ejemplo.com/contratos")
    api_key = st.text_input("Clave de API (Opcional)", value=api_config.get("API_KEY", ""), type="password", help="Si tu API requiere autenticación, introduce la clave aquí.")
    api_query = st.text_area("Parámetros de Consulta (Opcional)", value=api_config.get("QUERY", ""), height=150, help="Parámetros adicionales para la consulta.")
    
    if st.button("Guardar Configuración de API"):
        config["TOPICS"][ACTIVE_TOPIC]["DATA_SOURCE_CONFIG"]["API"] = {"BASE_URL": api_url, "API_KEY": api_key, "QUERY": api_query}
        save_config(config)
        st.session_state.config = config
        st.success("✅ Configuración de la API guardada.")

elif selected_source == "CSV":
    st.markdown("#### Gestión de Archivos Locales")
    st.markdown("Utiliza esta sección para subir nuevos archivos (CSV o Excel) y asignar uno como la fuente de datos de respaldo para este tema.")
    st.info("ℹ️ **Detección automática:** El sistema detectará automáticamente si tu archivo CSV usa comas (`,`) o punto y coma (`;`) como separador.")

    # Obtener el directorio de entrada específico para el tema
    topic_input_dir = get_topic_input_dir(ACTIVE_TOPIC)

    st.markdown(f"##### 📋 Archivos Disponibles en `archivos_entrada/{ACTIVE_TOPIC}`")
    input_files = list(topic_input_dir.glob("*.csv")) + list(topic_input_dir.glob("*.xlsx"))

    if input_files:
        # ... (código de visualización de archivos sin cambios)
        pass
    else:
        st.info(f"📂 No hay archivos de entrada disponibles en la carpeta del tema. Sube uno a continuación.")

    st.markdown("##### ⬆️ Subir Nuevo Archivo de Datos")
    uploaded_file = st.file_uploader("Selecciona un archivo CSV o Excel:", type=['csv', 'xlsx', 'xls'])

    if uploaded_file is not None:
        save_path = topic_input_dir / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Archivo '{uploaded_file.name}' subido exitosamente a `archivos_entrada/{ACTIVE_TOPIC}`.")
        st.rerun()

    st.markdown("##### 📌 Asignar Archivo de Datos para el Tema Activo")
    csv_config = data_source_config.get("CSV", {})
    current_csv_filename = csv_config.get("FILENAME", "")
    
    available_files_for_topic = [f.name for f in input_files] if input_files else []
    current_index = available_files_for_topic.index(current_csv_filename) if current_csv_filename in available_files_for_topic else 0

    new_csv_file = st.selectbox("Archivo de datos a usar para este tema:", options=available_files_for_topic, index=current_index)
    
    if st.button("Guardar Nombre de Archivo para Tema"):
        if new_csv_file:
            config["TOPICS"][ACTIVE_TOPIC]["DATA_SOURCE_CONFIG"]["CSV"] = {"FILENAME": new_csv_file}
            save_config(config)
            st.session_state.config = config
            st.success(f"✅ Archivo de datos '{new_csv_file}' asignado para el tema '{ACTIVE_TOPIC}'.")
        else:
            st.warning("No hay archivos disponibles para asignar.")

elif selected_source == "SQL":
    st.markdown("#### Configuración de la Bodega de Datos (SQL Server)")
    sql_config = data_source_config.get("SQL", {})
    
    host = st.text_input("Host", value=sql_config.get("HOST", "localhost"))
    port = st.text_input("Puerto", value=sql_config.get("PORT", "1433"))
    database = st.text_input("Base de Datos", value=sql_config.get("DATABASE", ""))
    username = st.text_input("Usuario", value=sql_config.get("USERNAME", ""))
    password = st.text_input("Contraseña", value=sql_config.get("PASSWORD", ""), type="password")
    driver = st.text_input("Driver ODBC de Microsoft SQL Server", value=sql_config.get("DRIVER", "ODBC Driver 17 for SQL Server"), help="El driver oficial de Microsoft para SQL Server. Búscalo en 'Orígenes de datos ODBC' en Windows.")
    query = st.text_area("Consulta SQL", value=sql_config.get("QUERY", "SELECT * FROM nombre_tabla"), height=200)

    if st.button("Guardar Configuración de Bodega de Datos"):
        config["TOPICS"][ACTIVE_TOPIC]["DATA_SOURCE_CONFIG"]["SQL"] = {
            "DB_TYPE": "mssql", "HOST": host, "PORT": port, "DATABASE": database,
            "USERNAME": username, "PASSWORD": password, "DRIVER": driver, "QUERY": query
        }
        save_config(config)
        st.session_state.config = config
        st.success("✅ Configuración de la Bodega de Datos guardada.")

st.markdown("---")

# --- 3. Configuración de Prefiltrado ---
st.markdown("### 2. Prefiltrado Dinámico de Datos (Opcional)")
st.info("Aplica filtros para reducir el volumen de datos antes del análisis. Si no defines reglas, se usarán todos los datos.")

if 'available_columns' not in st.session_state:
    st.session_state.available_columns = []

if st.button("Cargar / Actualizar Columnas de la Fuente"):
    with st.spinner("Obteniendo columnas..."):
        st.session_state.available_columns = obtener_columnas_fuente_datos()
        if not st.session_state.available_columns:
            st.warning("No se pudieron obtener las columnas. Verifica la configuración de la fuente de datos.")
        else:
            st.success(f"Se encontraron {len(st.session_state.available_columns)} columnas.")

if st.session_state.available_columns:
    st.success(f"Se encontraron {len(st.session_state.available_columns)} columnas.")
    
    prefiltrado_config = topic_config.get("PREFILTRADO_CONFIG", {"FILTROS_ACTIVOS": []})
    filtros = prefiltrado_config.get("FILTROS_ACTIVOS", [])
    
    # ... (código del data_editor sin cambios)
    edited_filtros = st.data_editor(
        pd.DataFrame(filtros) if filtros else pd.DataFrame([{"campo": "", "valores": ""}]),
        num_rows="dynamic", use_container_width=True,
        column_config={
            "campo": st.column_config.SelectboxColumn("Campo a Filtrar", options=st.session_state.available_columns, required=True),
            "valores": st.column_config.TextColumn("Valores a Incluir (separados por coma)", required=True)
        }
    )

    if st.button("💾 Guardar Reglas de Prefiltrado"):
        nuevos_filtros = []
        for _, row in edited_filtros.iterrows():
            campo = row['campo']
            valores_str = row['valores']
            if campo and valores_str:
                valores_list = [v.strip() for v in valores_str.split(',')]
                nuevos_filtros.append({"campo": campo, "valores": valores_list})
        
        config["TOPICS"][ACTIVE_TOPIC]["PREFILTRADO_CONFIG"] = {"FILTROS_ACTIVOS": nuevos_filtros}
        save_config(config)
        st.session_state.config = config
        st.success("✅ Reglas de prefiltrado guardadas exitosamente.")

st.markdown("---")

# --- 4. Ejecución y Verificación ---
st.markdown("### 3. Verificación y Carga de Datos")
st.markdown("Una vez configurado, puedes ejecutar el proceso para cargar, filtrar y guardar los datos . Este fichero será la entrada para el resto del sistema.")

if st.button("🚀 Generar Archivo de Datos Estandarizado", type="primary"):
    with st.spinner("Cargando, filtrando y guardando datos... Esta operación puede tardar unos minutos."):
        exito = cargar_datos_y_aplicar_filtros()
    
    if exito:
        st.success("¡Proceso completado!")
        st.markdown("Este archivo será la fuente de datos para todos los demás pasos del pipeline.")
        from utils import cargar_datos_preprocesados
        df_preview = cargar_datos_preprocesados(ACTIVE_TOPIC)
        if not df_preview.empty:
            st.markdown("**Vista previa de los datos procesados:**")
            st.dataframe(df_preview.head())
    else:
        st.error("El proceso falló. Revisa los mensajes de error en la consola o en los logs.")