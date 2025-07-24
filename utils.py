import streamlit as st
import pandas as pd
from pathlib import Path
import json
import subprocess
import sys
import shutil
import threading
import time
import psutil
import requests
from sodapy import Socrata
import os
from dotenv import load_dotenv

import re
import io
import logging
from datetime import datetime
import hashlib

# Carga las variables definidas en el archivo .env al entorno actual
load_dotenv()

# --- Constantes Centralizadas ---
BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / 'config.json'
INPUT_DIR = BASE_DIR / "archivos_entrada"
RESULTS_DIR = BASE_DIR / "resultados"
ASSETS_DIR = BASE_DIR / "assets"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
HISTORY_DIR = RESULTS_DIR / "historial_validaciones"

# --- Diccionario de Términos Simplificados ---
TERMINOS_AMIGABLES = {
    # Términos técnicos -> Términos amigables
    "fine-tuning": "entrenamiento especializado",
    "embeddings": "análisis de texto",
    "modelo": "sistema de clasificación",
    "pipeline": "proceso",
    "Similitud_Semantica_Max": "Relevancia (%)",
    "Metodo_Deteccion": "Encontrado por",
    "keywords": "palabras clave",
    "threshold": "umbral",
    "dataset": "conjunto de datos",
    "clasificador": "sistema clasificador",
    "validación": "revisión",
    "entrenamiento": "aprendizaje",
    "predicción": "clasificación",
    "métricas": "resultados",
    "accuracy": "precisión",
    "recall": "cobertura",
    "precision": "exactitud",
    "F1-score": "puntuación F1",
    "confianza": "certeza",
    "cache": "memoria temporal",
    "preprocessing": "preparación de datos",
    "tokenización": "división de texto",
    "lemmatización": "simplificación de palabras"
}

# --- Descripciones Amigables ---
DESCRIPCIONES_AMIGABLES = {
    "fine-tuning": "Proceso donde el sistema aprende específicamente sobre el tema de interés",
    "embeddings": "Representación matemática del texto que permite comparar similitudes",
    "similitud_semantica": "Qué tan relacionado está el contenido de un contrato con el tema buscado",
    "metodo_deteccion": "Cómo el sistema encontró este contrato (por palabras clave o análisis de contenido)",
    "validacion_humana": "Revisión por parte de un experto para confirmar si el contrato es relevante",
    "entrenamiento_modelo": "Proceso donde el sistema aprende de las validaciones del experto",
    "prediccion": "Clasificación automática que hace el sistema basado en lo aprendido",
    "confianza": "Qué tan seguro está el sistema de su clasificación (0-100%)",
    "precision": "De todos los contratos que el sistema dice que son relevantes, cuántos realmente lo son",
    "recall": "De todos los contratos relevantes que existen, cuántos logró encontrar el sistema",
    "f1_score": "Medida que combina precisión y cobertura en un solo número"
}

# Crear directorios si no existen
INPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)

# --- Sistema de Logging Unificado ---

def setup_logging(script_name: str, topic: str = None) -> logging.Logger:
    """
    Configura un logger unificado para el sistema.
    
    Args:
        script_name: Nombre del script (ej: 'preprocesamiento', 'entrenamiento')
        topic: Tema activo (opcional)
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    
    # Evitar duplicar handlers
    if logger.handlers:
        return logger
    
    # Crear formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (con rotación básica)
    if topic:  # Fixed semicolon here
        log_filename = f"{script_name}_{topic}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    else:
        log_filename = f"{script_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    # Create a 'logs' subdirectory within RESULTS_DIR if it doesn't exist
    logs_subdir = RESULTS_DIR / "logs"
    logs_subdir.mkdir(exist_ok=True)

    log_file = logs_subdir / log_filename
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# --- Funciones de Usabilidad ---

def get_termino_amigable(termino_tecnico: str) -> str:
    """Convierte un término técnico a uno más amigable para el usuario."""
    return TERMINOS_AMIGABLES.get(termino_tecnico.lower(), termino_tecnico)

def get_descripcion_amigable(termino_tecnico: str) -> str:
    """Obtiene una descripción amigable de un término técnico."""
    return DESCRIPCIONES_AMIGABLES.get(termino_tecnico.lower(), "")

def crear_tooltip(texto: str, descripcion: str) -> str:
    """Crea un tooltip HTML con descripción amigable."""
    return f'<span title="{descripcion}">{texto}</span>'

def mostrar_ayuda_contextual(termino: str) -> str:
    """Genera ayuda contextual para un término específico."""
    descripcion = get_descripcion_amigable(termino)
    if descripcion:
        return f"ℹ️ {descripcion}"
    return ""

def simplificar_columna_nombre(nombre_columna: str) -> str:
    """Simplifica nombres de columnas técnicas para la interfaz."""
    mapeo_columnas = {
        "Similitud_Semantica_Max": "Relevancia (%)",
        "Metodo_Deteccion": "Encontrado por",
        "Es_Ciberseguridad_Validado": "¿Es relevante?",
        "Confianza_Prediccion": "Certeza del sistema",
        "Precision_Keyword": "Efectividad de palabra clave",
        "Recall_Keyword": "Cobertura de palabra clave"
    }
    return mapeo_columnas.get(nombre_columna, nombre_columna)

def formatear_porcentaje(valor: float) -> str:
    """Formatea un valor decimal como porcentaje amigable."""
    if pd.isna(valor):
        return "N/A"
    return f"{valor:.1%}"

def formatear_confianza(valor: float) -> str:
    """Formatea valores de confianza con indicadores visuales."""
    if pd.isna(valor):
        return "N/A"
    
    porcentaje = valor * 100
    if porcentaje >= 80:
        return f"🟢 {porcentaje:.0f}% (Alta)"
    elif porcentaje >= 60:
        return f"🟡 {porcentaje:.0f}% (Media)"
    else:
        return f"🔴 {porcentaje:.0f}% (Baja)"

# --- INICIO DE MODIFICACIÓN: Limpieza de Keywords ---
def limpiar_termino_busqueda(termino: str) -> str:
    """
    Limpia un término de búsqueda (keyword) para consistencia.
    Convierte a minúsculas, elimina tildes y espacios extra.
    """
    if not isinstance(termino, str):
        return ""
    texto = termino.lower().strip()
    # Reemplazo de vocales con tildes
    texto = texto.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    return texto
# --- FIN DE MODIFICACIÓN ---

# --- Funciones de Procesamiento de Texto ---

def limpiar_texto(texto: str, nlp_model) -> str:
    """
    Limpia y lematiza el texto para que coincida con el preprocesamiento de entrenamiento.
    Reduce las palabras a su forma raíz (lema) para una mejor coincidencia.
    Ej: 'servicios de consultorías' -> 'servicio de consultoría'
    """
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = texto.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    texto = re.sub(r'[^\w\s-]', '', texto)
    texto = texto.strip()
    doc = nlp_model(texto)
    return " ".join([token.lemma_ for token in doc])

def preparar_texto_para_modelo(df: pd.DataFrame, text_columns: list, nlp_model) -> pd.DataFrame:
    """
    Centraliza el proceso de combinar, limpiar y lematizar texto para el modelo.
    Asegura que el preprocesamiento sea idéntico en todas las fases del pipeline.

    Args:
        df: DataFrame de entrada.
        text_columns: Lista de columnas de texto a combinar.
        nlp_model: Modelo de spaCy cargado.

    Returns:
        DataFrame con una nueva columna 'texto_limpio'.
    """
    # --- INICIO DE MODIFICACIÓN: Añadir logging y robustez ---
    logger = setup_logging("text_preparator", get_active_topic())
    logger.info(f"Iniciando preparación de texto. Se intentarán combinar las siguientes columnas: {text_columns}")

    df_copy = df.copy()
    
    # 1. Combinar columnas de texto
    existing_cols = [col for col in text_columns if col in df_copy.columns]
    missing_cols = [col for col in text_columns if col not in df_copy.columns]

    if missing_cols:
        logger.warning(f"Las siguientes columnas configuradas no se encontraron y serán ignoradas: {missing_cols}")
    
    if not existing_cols:
        logger.error("¡Error crítico! Ninguna de las columnas de texto configuradas existe en los datos. La columna 'texto_limpio' estará vacía.")
        df_copy['texto_combinado'] = ""
    else:
        logger.info(f"Columnas que se combinarán para el análisis: {existing_cols}")
        df_copy['texto_combinado'] = df_copy[existing_cols].fillna('').astype(str).agg(' '.join, axis=1)
    
    # 2. Limpieza básica (rápida, antes de spaCy)
    def limpieza_basica(texto: str) -> str:
        if not isinstance(texto, str): return ""
        texto = texto.lower()
        texto = texto.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        texto = re.sub(r'[^\w\s-]', '', texto) # Eliminar caracteres no alfanuméricos excepto guiones
        return texto.strip()
    
    df_copy['texto_preparado'] = df_copy['texto_combinado'].apply(limpieza_basica)
    
    # 3. Lematización eficiente con spaCy
    texts_to_process = df_copy['texto_preparado'].astype(str).tolist()
    docs = nlp_model.pipe(texts_to_process, batch_size=128, n_process=-1) 
    df_copy['texto_limpio'] = [" ".join([token.lemma_ for token in doc]) for doc in docs]
    
    # 4. Limpiar columnas intermedias
    df_copy.drop(columns=['texto_combinado', 'texto_preparado'], inplace=True, errors='ignore')
    
    return df_copy

# --- Funciones de Hashing y Gestión de Historial ---

def generar_hash_contrato(df: pd.DataFrame, columnas_a_hashear: list) -> pd.DataFrame:
    """
    Genera un hash SHA256 único para cada fila basado en las columnas especificadas.

    Args:
        df: DataFrame de entrada.
        columnas_a_hashear: Lista de columnas para concatenar y hashear.

    Returns:
        DataFrame con una nueva columna 'hash_contrato'.
    """
    if df.empty or not columnas_a_hashear:
        df['hash_contrato'] = None
        return df

    def calcular_hash(row):
        # Concatenar los valores de las columnas, manejando nulos
        texto_a_hashear = "".join(str(row.get(col, '')) for col in columnas_a_hashear)
        return hashlib.sha256(texto_a_hashear.encode('utf-8')).hexdigest()

    df['hash_contrato'] = df.apply(calcular_hash, axis=1)
    return df

def get_historial_hashes_path(topic: str) -> Path:
    """Devuelve la ruta al archivo que contiene los hashes ya validados."""
    return RESULTS_DIR / f"hashes_validados_{topic}.csv"

def cargar_hashes_validados(topic: str) -> set:
    """
    Carga el conjunto de hashes que ya han sido validados por un humano.

    Args:
        topic: El tema activo.

    Returns:
        Un conjunto (set) de hashes validados.
    """
    historial_path = get_historial_hashes_path(topic)
    if not historial_path.exists():
        return set()
    try:
        df_hashes = pd.read_csv(historial_path)
        return set(df_hashes['hash_contrato'].tolist())
    except (pd.errors.EmptyDataError, KeyError):
        return set()

def anadir_hashes_validados(topic: str, nuevos_hashes: list):
    """
    Añade una lista de nuevos hashes validados al archivo de historial.

    Args:
        topic: El tema activo.
        nuevos_hashes: Una lista de los nuevos hashes a añadir.
    """
    if not nuevos_hashes:
        return

    historial_path = get_historial_hashes_path(topic)
    
    # Cargar hashes existentes para evitar duplicados
    hashes_existentes = cargar_hashes_validados(topic)
    
    # Filtrar para añadir solo los que no existen
    hashes_a_anadir = [h for h in nuevos_hashes if h not in hashes_existentes]
    
    if not hashes_a_anadir:
        return

    df_nuevos = pd.DataFrame(hashes_a_anadir, columns=['hash_contrato'])
    
    # Usar modo 'a' (append) y no escribir el header si el archivo ya existe
    header = not historial_path.exists()
    df_nuevos.to_csv(historial_path, mode='a', header=header, index=False)


def consolidar_validaciones_historicas(topic: str) -> pd.DataFrame:
    """
    Consolida todas las validaciones de un tema desde el historial para obtener la versión más reciente de cada una.

    Args:
        topic: El tema activo.

    Returns:
        Un DataFrame con las validaciones más actuales, conteniendo ['hash_contrato', 'validacion'].
    """
    logger = setup_logging("consolidator", topic)
    history_dir = RESULTS_DIR / "historial_validaciones"
    validation_col_name = f'Es_{topic.capitalize()}_Validado'

    if not history_dir.exists():
        logger.warning(f"El directorio de historial '{history_dir}' no existe. No hay nada que consolidar.")
        return pd.DataFrame(columns=['hash_contrato', validation_col_name])

    all_validations = []
    # Regex para extraer timestamp del nombre del archivo
    timestamp_regex = re.compile(r"_(\d{8}_\d{6})")

    # 1. Leer todos los archivos de historial
    for a_file in history_dir.glob(f"validacion_{topic}_*.xlsx"):
        match = timestamp_regex.search(a_file.name)
        if not match:
            continue
        
        timestamp_str = match.group(1)
        validation_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        
        try:
            df = pd.read_excel(a_file, engine='openpyxl')
            df['fecha_validacion'] = validation_time
            all_validations.append(df)
        except Exception as e:
            logger.error(f"No se pudo leer el archivo de historial '{a_file.name}': {e}")

    if not all_validations:
        logger.info("No se encontraron archivos de validación en el historial.")
        return pd.DataFrame(columns=['hash_contrato', validation_col_name])

    # 2. Combinar todo en un único DataFrame
    df_combined = pd.concat(all_validations, ignore_index=True)

    # 3. Asegurarse de que las columnas necesarias existan
    if 'hash_contrato' not in df_combined.columns or validation_col_name not in df_combined.columns:
        logger.error("Los archivos de historial no tienen las columnas 'hash_contrato' o la columna de validación.")
        return pd.DataFrame(columns=['hash_contrato', validation_col_name])

    # 4. Filtrar solo las filas que tienen una validación real
    df_combined = df_combined[df_combined[validation_col_name].isin(['SI', 'NO'])].copy()

    # 5. Ordenar por fecha para que la última validación de un mismo contrato quede al final
    df_combined.sort_values(by='fecha_validacion', ascending=True, inplace=True)

    # 6. Eliminar duplicados por hash, manteniendo solo la última entrada (la más reciente)
    df_final = df_combined.drop_duplicates(subset=['hash_contrato'], keep='last')

    logger.info(f"Consolidación completa. Se encontraron {len(df_final)} validaciones únicas y actualizadas.")

    return df_final[['hash_contrato', validation_col_name]]
# --- Funciones de Configuración (Ejemplos) ---

def load_config():
    """Carga el archivo de configuración principal con validación."""
    if not CONFIG_FILE.exists():
        default_config = {"ACTIVE_TOPIC": "", "TOPICS": {}}
        return default_config
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Validar la configuración cargada
        is_valid, message = validate_config(config_data)
        if not is_valid:
            print(f"⚠️  Advertencia: Configuración inválida: {message}")
            print("Se usará configuración por defecto.")
            return {"ACTIVE_TOPIC": "", "TOPICS": {}}
        
        return config_data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"❌ Error al cargar configuración: {e}")
        return {"ACTIVE_TOPIC": "", "TOPICS": {}}

def validate_config(config_data):
    """Valida la estructura del archivo de configuración."""
    required_keys = ["ACTIVE_TOPIC", "TOPICS"]
    
    for key in required_keys:
        if key not in config_data:
            return False, f"Falta la clave requerida: {key}"
    
    if not isinstance(config_data["TOPICS"], dict):
        return False, "TOPICS debe ser un diccionario"
    
    active_topic = config_data.get("ACTIVE_TOPIC")
    if active_topic and active_topic not in config_data["TOPICS"]:
        return False, f"El tema activo '{active_topic}' no existe en TOPICS"
    
    # Validar estructura de cada tema
    for topic_name, topic_config in config_data["TOPICS"].items():
        if not isinstance(topic_config, dict):
            return False, f"La configuración del tema '{topic_name}' debe ser un diccionario"
        
        required_topic_keys = ["DATA_SOURCE_CONFIG", "TEXT_COLUMNS_TO_COMBINE", "FILTRADO_UNSPSC"]
        for key in required_topic_keys:
            if key not in topic_config:
                return False, f"Falta la clave '{key}' en el tema '{topic_name}'"
        
        # Validar FILTRADO_UNSPSC
        filtrado = topic_config.get("FILTRADO_UNSPSC", {})
        if not isinstance(filtrado, dict) or "CODIGOS_DE_INTERES" not in filtrado:
            return False, f"FILTRADO_UNSPSC inválido en el tema '{topic_name}'"
        
        # Validar TEXT_COLUMNS_TO_COMBINE
        if not isinstance(topic_config["TEXT_COLUMNS_TO_COMBINE"], list):
            return False, f"TEXT_COLUMNS_TO_COMBINE debe ser una lista en el tema '{topic_name}'"
    
    return True, "Configuración válida"

def save_config(config_data):
    """Guarda los datos en el archivo de configuración después de validar."""
    is_valid, message = validate_config(config_data)
    if not is_valid:
        raise ValueError(f"Configuración inválida: {message}")
    
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

def get_active_topic():
    """Obtiene el tema activo desde la configuración."""
    config = load_config()
    return config.get("ACTIVE_TOPIC", "")

def get_active_topic_config():
    """Obtiene la configuración completa para el tema activo."""
    config = load_config()
    active_topic = config.get("ACTIVE_TOPIC")
    if not active_topic:
        return None
    return config.get("TOPICS", {}).get(active_topic)

# --- Funciones de Carga de Datos ---

def cargar_datos_soda(soda_config: dict) -> pd.DataFrame | None:
    """
    Carga datos desde una API de Socrata (Datos Abiertos).
    """
    logger = setup_logging("soda_loader")
    domain = soda_config.get("DOMAIN")
    dataset_id = soda_config.get("DATASET_ID")
    select_clause = soda_config.get("SELECT_CLAUSE")
    where_clause = soda_config.get("WHERE_CLAUSE")

    if not domain or not dataset_id:
        logger.error("El dominio y el ID del dataset son requeridos para la API de Socrata.")
        st.error("El dominio y el ID del dataset son requeridos.")
        return None

    usuario = os.getenv("DATOS_GOV_USER")
    password = os.getenv("DATOS_GOV_PASS")
    app_token = os.getenv("DATOS_GOV_TOKEN")

    if not all([usuario, password, app_token]):
        logger.error("Credenciales de Socrata no encontradas en .env.")
        st.error("Asegúrate de definir DATOS_GOV_USER, DATOS_GOV_PASS, y DATOS_GOV_TOKEN en tu archivo .env.")
        return None

    try:
        client = Socrata(domain, app_token, username=usuario, password=password, timeout=600)
        logger.info(f"Conectando a {domain} para el dataset {dataset_id}...")
        
        results = client.get(dataset_id, select=select_clause, where=where_clause, limit=2000000) # Límite alto
        
        df = pd.DataFrame.from_records(results)
        logger.info(f"Se obtuvieron {len(df)} registros desde la API de Socrata.")
        st.success(f"✅ Se obtuvieron {len(df)} registros desde la API de Datos Abiertos.")
        return df
    except Exception as e:
        logger.error(f"Error al cargar datos desde Socrata: {e}")
        st.error(f"Error al conectar con la API de Datos Abiertos: {e}")
        return None

def get_file_columns(file_path_str: str) -> list[str]:
    """
    Lee las columnas de un archivo CSV o Excel de forma eficiente.
    Cachea el resultado para evitar lecturas repetidas del disco.
    
    Args:
        file_path_str: La ruta del archivo como string.
    
    Returns:
        Una lista de nombres de columnas o una lista vacía si hay un error.
    """
    logger = setup_logging("utils")
    file_path = Path(file_path_str)
    if not file_path.exists():
        logger.warning(f"Intento de leer columnas de un archivo inexistente: {file_path_str}")
        return []
    try:
        if file_path.suffix.lower() == '.csv':
            # Usar sep=None y engine='python' para auto-detectar el separador
            return pd.read_csv(file_path, nrows=0, encoding='utf-8', on_bad_lines='skip', sep=None, engine='python').columns.tolist()
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, nrows=0, engine='openpyxl').columns.tolist()
        elif file_path.suffix.lower() == '.parquet':
            return pd.read_parquet(file_path).columns.tolist()
        else:
            logger.warning(f"Formato de archivo no soportado para leer columnas: {file_path.suffix}")
            return []
    except Exception as e:
        logger.error(f"Error al leer columnas de {file_path}: {e}")
        return []

def obtener_columnas_fuente_datos() -> list[str]:
    """
    Obtiene solo los nombres de las columnas de la fuente de datos activa.
    Es una operación ligera que no carga todo el dataset.

    Returns:
        Una lista con los nombres de las columnas o una lista vacía si hay un error.
    """
    topic_config = get_active_topic_config()
    if not topic_config:
        st.error("No hay un tema activo configurado.")
        return []

    source_config = topic_config.get("DATA_SOURCE_CONFIG", {})
    active_source = source_config.get("ACTIVE_SOURCE", "CSV")

    try:
        if active_source == "API_SODA":
            soda_config = source_config.get("API_SODA", {})
            df = cargar_datos_soda(soda_config)
            if df is not None:
                return df.columns.tolist()
            return []
        elif active_source == "API":
            api_details = source_config.get("API", {})
            url = api_details.get("BASE_URL")
            if not url:
                st.error("La URL de la API no está configurada para este tema.")
                return []
            params = {"$limit": 1}
            api_query = api_details.get("QUERY")
            if api_query:
                params["$query"] = api_query

            headers = {"Authorization": f"Bearer {api_details.get('API_KEY')}"} if api_details.get('API_KEY') else {}
            response = requests.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and data:
                return list(data[0].keys())
            elif isinstance(data, dict) and 'results' in data and data['results']:
                return list(data['results'][0].keys())
            else:
                st.warning("La respuesta de la API no tiene el formato esperado.")
                return []

        elif active_source == "SQL":
            try:
                import sqlalchemy
            except ImportError:
                st.error("SQLAlchemy no está instalado. Instálalo con: pip install sqlalchemy")
                return []
            sql_details = source_config.get("SQL", {})
            query = f"SELECT * FROM ({sql_details.get('QUERY', 'SELECT 1')}) AS subquery LIMIT 1"
            connect_args = {'connect_timeout': sql_details.get("CONNECTION_TIMEOUT", 10)} if sql_details.get("DB_TYPE") == "postgresql" else {}
            engine = sqlalchemy.create_engine(get_sql_connection_string(sql_details), connect_args=connect_args)
            with engine.connect() as connection:
                return pd.read_sql(query, connection).columns.tolist()

        elif active_source == "LOCAL_FILE":
            local_file_details = source_config.get("LOCAL_FILE", {})
            filename = local_file_details.get("FILENAME", "")
            if not filename:
                st.warning("No se ha configurado un archivo de datos para este tema.")
                return []

            active_topic = get_active_topic()
            if not active_topic:
                st.error("No hay tema activo seleccionado.")
                return []

            local_file_path = get_topic_input_dir(active_topic) / filename
            if not local_file_path.exists():
                st.error(f"El archivo no se encuentra en: {local_file_path}")
                return []

            if local_file_path.suffix.lower() == '.csv':
                return pd.read_csv(local_file_path, nrows=0, sep=None, engine='python').columns.tolist()
            elif local_file_path.suffix.lower() in ['.xlsx', '.xls']:
                return pd.read_excel(local_file_path, nrows=0, engine='openpyxl').columns.tolist()
            else:
                st.error(f"Formato de archivo no soportado: {local_file_path.suffix}. Use CSV o Excel.")
                return []

        elif active_source == "PARQUET":
            parquet_details = source_config.get("PARQUET", {})
            filename = parquet_details.get("FILENAME", "")
            if not filename:
                st.warning("No se ha configurado un archivo Parquet para este tema.")
                return []

            active_topic = get_active_topic()
            if not active_topic:
                st.error("No hay tema activo seleccionado.")
                return []

            parquet_path = get_topic_results_dir(active_topic) / filename
            if not parquet_path.exists():
                st.error(f"El archivo Parquet no se encuentra en: {parquet_path}")
                return []
            
            return pd.read_parquet(parquet_path).columns.tolist()

    except requests.exceptions.RequestException as e:
        st.error(f"Error de red al conectar con la API: {e}")
        return []
    except Exception as e:
        if 'sqlalchemy' in str(type(e).__module__):
            st.error(f"Error al conectar con la base de datos SQL: {e}")
        else:
            st.error(f"Error al obtener columnas de la fuente '{active_source}': {e}")
        return []
    
    return []

def get_sql_connection_string(sql_config: dict) -> str:
    """Construye la cadena de conexión de SQLAlchemy a partir de la configuración."""
    db_type = sql_config.get("DB_TYPE", "postgresql").lower()
    user = sql_config.get("USERNAME")
    password = sql_config.get("PASSWORD")
    host = sql_config.get("HOST")
    port = sql_config.get("PORT")
    database = sql_config.get("DATABASE")

    if db_type == "mssql" or db_type == "sqlserver": # Mantener compatibilidad con 'sqlserver'
        # Obtener el driver desde la configuración, con un valor por defecto para compatibilidad
        driver = sql_config.get("DRIVER", "ODBC Driver 17 for SQL Server")
        # Reemplazar espacios con '+' para la URL de conexión
        driver_url_encoded = driver.replace(' ', '+')
        # El dialecto mssql+pyodbc es el estándar actual. Requiere `pip install pyodbc`.
        return f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver={driver_url_encoded}"

    return f"{db_type}://{user}:{password}@{host}:{port}/{database}"


def cargar_fuente_de_datos(active_topic: str, topic_config: dict) -> dict:
    """
    Carga datos desde la fuente configurada (CSV/Excel, SQL, SODA) y devuelve un DataFrame.

    Args:
        active_topic: El nombre del tema activo.
        topic_config: La configuración del tema activo.

    Returns:
        Un diccionario con:
        - success (bool): True si la carga fue exitosa.
        - message (str): Mensaje informativo sobre el resultado.
        - dataframe (pd.DataFrame | None): El DataFrame cargado o None si falló.
        - rows_loaded (int): Número de filas cargadas.
        - source_used (str): La fuente de datos que se utilizó.
    """
    logger = setup_logging("data_loader", active_topic)
    source_config = topic_config.get("DATA_SOURCE_CONFIG", {})
    active_source = source_config.get("ACTIVE_SOURCE", "LOCAL_FILE")
    
    df = None
    message = ""
    source_used = ""

    # 1. Cargar desde archivo local (CSV/Excel)
    if active_source == "LOCAL_FILE":
        source_used = "Archivo Local"
        local_file_config = source_config.get("LOCAL_FILE", {})
        file_name = local_file_config.get("FILENAME")
        if not file_name:
            return {"success": False, "message": "No se ha configurado un nombre de archivo en LOCAL_FILE.", "dataframe": None, "rows_loaded": 0, "source_used": source_used}

        file_path = get_topic_input_dir(active_topic) / file_name
        if not file_path.exists():
            return {"success": False, "message": f"El archivo '{file_name}' no se encuentra en la carpeta de entrada del tema.", "dataframe": None, "rows_loaded": 0, "source_used": source_used}

        try:
            logger.info(f"Cargando datos desde el archivo local: {file_path}")
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                return {"success": False, "message": f"Formato de archivo no soportado: {file_path.suffix}", "dataframe": None, "rows_loaded": 0, "source_used": source_used}
            
            message = f"Se cargaron {len(df)} filas desde el archivo '{file_name}'."
            logger.info(message)
            return {"success": True, "message": message, "dataframe": df, "rows_loaded": len(df), "source_used": source_used}

        except Exception as e:
            logger.error(f"Error al leer el archivo local '{file_name}': {e}")
            return {"success": False, "message": f"Error al leer el archivo: {e}", "dataframe": None, "rows_loaded": 0, "source_used": source_used}

    # 2. Cargar desde SQL
    elif active_source == "SQL":
        source_used = "Base de Datos SQL"
        sql_config = source_config.get("SQL", {})
        if not sql_config.get("HOST"):
            return {"success": False, "message": "La configuración de SQL (HOST) no está completa.", "dataframe": None, "rows_loaded": 0, "source_used": source_used}
        
        try:
            import sqlalchemy
            logger.info("Cargando datos desde la base de datos SQL...")
            engine = sqlalchemy.create_engine(get_sql_connection_string(sql_config))
            with engine.connect() as connection:
                df = pd.read_sql(sql_config["QUERY"], connection)
            
            message = f"Se cargaron {len(df)} filas desde la consulta SQL."
            logger.info(message)
            return {"success": True, "message": message, "dataframe": df, "rows_loaded": len(df), "source_used": source_used}

        except ImportError:
            message = "La librería 'sqlalchemy' no está instalada. No se puede conectar a SQL."
            logger.error(message)
            return {"success": False, "message": message, "dataframe": None, "rows_loaded": 0, "source_used": source_used}
        except Exception as e:
            logger.error(f"Error al cargar datos desde SQL: {e}")
            return {"success": False, "message": f"Error en la conexión SQL: {e}", "dataframe": None, "rows_loaded": 0, "source_used": source_used}

    # 3. Cargar desde API SODA
    elif active_source == "API_SODA":
        source_used = "API de Datos Abiertos (SODA)"
        soda_config = source_config.get("API_SODA", {})
        df = cargar_datos_soda(soda_config) # Reutilizamos la función existente
        if df is not None:
            message = f"Se cargaron {len(df)} filas desde la API de Socrata."
            return {"success": True, "message": message, "dataframe": df, "rows_loaded": len(df), "source_used": source_used}
        else:
            return {"success": False, "message": "Falló la carga de datos desde la API de Socrata.", "dataframe": None, "rows_loaded": 0, "source_used": source_used}
            
    else:
        return {"success": False, "message": f"Fuente de datos '{active_source}' no reconocida.", "dataframe": None, "rows_loaded": 0, "source_used": "Desconocida"}


def cargar_datos_y_aplicar_filtros() -> bool:
    """
    Carga datos siguiendo la jerarquía API -> SQL -> Archivo Local, aplica filtros
    y guarda el resultado en un archivo Parquet estandarizado.

    Returns:
        True si el proceso es exitoso, False en caso contrario.
    """
    active_topic = get_active_topic()
    if not active_topic:
        st.error("No hay un tema activo configurado.")
        return False

    topic_config = get_active_topic_config()
    if not topic_config:
        st.error(f"No se encontró la configuración para el tema '{active_topic}'.")
        return False

    # 1. Cargar datos usando la nueva función centralizada
    resultado_carga = cargar_fuente_de_datos(active_topic, topic_config)

    if not resultado_carga["success"]:
        st.error(f"❌ Error al cargar datos: {resultado_carga['message']}")
        return False

    df = resultado_carga["dataframe"]
    st.success(f"✅ {resultado_carga['message']} (Fuente: {resultado_carga['source_used']})")

    if df is None or df.empty:
        st.warning("El DataFrame está vacío después de la carga. No se puede continuar.")
        return False

    # --- INICIO DE MODIFICACIÓN: Generación de Hash ---
    logger = setup_logging("data_loader", active_topic)
    logger.info("Generando hashes únicos para cada contrato...")

    posibles_nombres_columnas = ["objeto_del_contrato", "Objeto del Contrato", "Objeto Contrato"]
    columna_texto = None

    # Buscar la columna que coincida con algún nombre posible
    for nombre in posibles_nombres_columnas:
        if nombre in df.columns:
            columna_texto = nombre
            break

    if not columna_texto:
        logger.error("No se han configurado las 'TEXT_COLUMNS_TO_COMBINE' en config.json para el hash.")
        st.error("Error de configuración: Faltan las columnas de texto para generar el identificador único.")
        return False

    # Generar el hash utilizando la columna encontrada
    df = generar_hash_contrato(df, [columna_texto])
    logger.info("Hashes generados exitosamente.")
    # --- FIN DE MODIFICACIÓN ---




    # 2. Aplicar filtros (si están configurados)
    logger.info("Revisando configuración de prefiltrado...")
    initial_rows = len(df)


    # 3. Guardar en Parquet
    output_parquet_path = get_preprocessed_data_path(active_topic)
    try:
        # Asegurar que todas las columnas de tipo 'object' se traten como string
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
            
        df.to_parquet(output_parquet_path, index=False)
        
        final_rows = len(df)
        logger.info(f"Datos procesados y guardados en '{output_parquet_path.name}'. Filas iniciales: {initial_rows}, Filas finales: {final_rows}.")
        st.success(f"✅ Proceso completado. Se guardaron {final_rows} filas en el archivo Parquet.")
        
    except Exception as e:
        logger.error(f"Error al guardar el archivo Parquet: {e}")
        st.error(f"❌ Error al guardar el archivo Parquet estandarizado: {e}")
        return False

    return True
    """
    Carga datos siguiendo la jerarquía API -> SQL -> Archivo Local, aplica filtros
    y guarda el resultado en un archivo Parquet estandarizado.

    Returns:
        True si el proceso es exitoso, False en caso contrario.
    """
    active_topic = get_active_topic()
    if not active_topic:
        st.error("No hay un tema activo configurado.")
        return False

    topic_config = get_active_topic_config()
    if not topic_config:
        st.error(f"No se encontró la configuración para el tema '{active_topic}'.")
        return False

    # 1. Cargar datos usando la nueva función centralizada
    resultado_carga = cargar_fuente_de_datos(active_topic, topic_config)

    if not resultado_carga["success"]:
        st.error(f"❌ Error al cargar datos: {resultado_carga['message']}")
        return False

    df = resultado_carga["dataframe"]
    st.success(f"✅ {resultado_carga['message']} (Fuente: {resultado_carga['source_used']})")

    if df is None or df.empty:
        st.warning("El DataFrame está vacío después de la carga. No se puede continuar.")
        return False

    # --- INICIO DE MODIFICACIÓN: Generación de Hash ---
    logger.info("Generando hashes únicos para cada contrato...")
    text_columns = topic_config.get("TEXT_COLUMNS_TO_COMBINE", [])
    if not text_columns:
        logger.error("No se han configurado las 'TEXT_COLUMNS_TO_COMBINE' en config.json para el hash.")
        st.error("Error de configuración: Faltan las columnas de texto para generar el identificador único.")
        return False
    df = generar_hash_contrato(df, text_columns)
    logger.info("Hashes generados exitosamente.")
    # --- FIN DE MODIFICACIÓN ---

    # 2. Aplicar filtros (si están configurados)
    logger.info("Revisando configuración de prefiltrado...")
    initial_rows = len(df)


    # 3. Guardar en Parquet
    output_parquet_path = get_preprocessed_data_path(active_topic)
    try:
        # Asegurar que todas las columnas de tipo 'object' se traten como string
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
            
        df.to_parquet(output_parquet_path, index=False)
        
        final_rows = len(df)
        logger.info(f"Datos procesados y guardados en '{output_parquet_path.name}'. Filas iniciales: {initial_rows}, Filas finales: {final_rows}.")
        st.success(f"✅ Proceso completado. Se guardaron {final_rows} filas en el archivo Parquet.")
        
    except Exception as e:
        logger.error(f"Error al guardar el archivo Parquet: {e}")
        st.error(f"❌ Error al guardar el archivo Parquet estandarizado: {e}")
        return False

    return True


def cargar_datos_preprocesados(topic_name: str) -> pd.DataFrame:
    """
    Carga los datos preprocesados desde el archivo Parquet estandarizado.
    para un tema específico.

    Args:
        topic_name: El nombre del tema activo.
        
    Returns:
        Un DataFrame de pandas con los datos, o un DataFrame vacío si no se encuentra el archivo.
    """
    parquet_path = get_preprocessed_data_path(topic_name)
    if not parquet_path.exists():
        logging.error(f"El archivo de datos preprocesados '{parquet_path.name}' no fue encontrado. Ejecute primero el paso de generación de candidatos.")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(parquet_path)
        logging.info(f"Datos cargados exitosamente desde '{parquet_path.name}' ({len(df)} filas).")
        return df
    except Exception as e:
        logging.error(f"Error al leer el archivo Parquet '{parquet_path}': {e}")
        return pd.DataFrame()

# --- Funciones de Rutas Consistentes ---

def get_topic_input_dir(topic: str) -> Path:
    """Devuelve la ruta del directorio de entrada para un tema y se asegura de que exista."""
    if not topic:
        # Si no hay tema, devuelve el directorio base para evitar errores.
        return INPUT_DIR
    topic_dir = INPUT_DIR / topic
    topic_dir.mkdir(parents=True, exist_ok=True)
    return topic_dir

def get_topic_results_dir(topic: str) -> Path:
    """Devuelve la ruta del directorio de resultados para un tema y se asegura de que exista."""
    if not topic:
        # Fallback crucial para evitar errores si el tema está vacío o no definido.
        return RESULTS_DIR
    topic_dir = RESULTS_DIR / topic
    topic_dir.mkdir(parents=True, exist_ok=True)
    return topic_dir

def get_topic_logs_dir(topic: str) -> Path:
    """Devuelve la ruta de la carpeta de logs para un tema y se asegura de que exista."""
    logs_dir = get_topic_results_dir(topic) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir

def get_topic_history_dir(topic: str) -> Path:
    """Devuelve la ruta del directorio de historial de validaciones para un tema."""
    history_dir = get_topic_results_dir(topic) / "historial_validaciones"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir

def get_keywords_file_path(topic: str) -> Path:
    """Devuelve la ruta del archivo de keywords para un tema específico."""
    return get_topic_input_dir(topic) / "keywords.xlsx"

def get_exclusion_file_path(topic: str) -> Path:
    """Devuelve la ruta del archivo de exclusiones para un tema específico."""
    return get_topic_input_dir(topic) / "exclusion_words.xlsx"

def get_preprocessed_data_path(topic: str) -> Path:
    """Devuelve la ruta del archivo de datos preprocesados para un tema específico."""
    return get_topic_results_dir(topic) / "datos_preprocesados.parquet"

def get_classifier_model_path(topic: str) -> Path:
    """Devuelve la ruta del modelo clasificador entrenado para un tema específico."""
    return get_topic_results_dir(topic) / "clasificador_v1.joblib"

def get_predictions_path(topic: str, format: str = 'csv') -> Path:
    """Devuelve la ruta del archivo de predicciones para un tema (CSV o Excel)."""
    return get_topic_results_dir(topic) / f"predicciones.{format}"

def get_finetuning_dataset_path(topic: str) -> Path:
    """Devuelve la ruta del dataset de fine-tuning para un tema específico."""
    return get_topic_results_dir(topic) / "finetuning_dataset.csv"

def get_human_review_file_path(topic: str) -> Path:
    """Devuelve la ruta del archivo de revisión humana para un tema específico."""
    return get_topic_results_dir(topic) / "contratos_para_revision_humana.xlsx"

def get_active_review_file_path(topic: str) -> Path:
    """Devuelve la ruta del archivo de revisión activa para un tema específico."""
    return get_topic_results_dir(topic) / "revision_activa.xlsx"

def get_validated_hashes_path(topic: str) -> Path:
    """Devuelve la ruta al archivo que contiene los hashes ya validados."""
    return get_topic_results_dir(topic) / "hashes_validados.csv"

def get_finetuned_model_path(topic: str) -> Path:
    """Devuelve la ruta de la carpeta del modelo experto afinado."""
    model_dir = get_topic_results_dir(topic) / "modelos_afinados" / "experto-v1"
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    return model_dir

def get_cache_file_path(topic: str, cache_type: str = "examples") -> Path:
    """Devuelve la ruta para diferentes tipos de archivos de caché (ej. 'embeddings', 'tfidf')."""
    return get_topic_results_dir(topic) / f"cache_{cache_type}.joblib"

def get_report_path(topic: str, report_name: str) -> Path:
    """Devuelve la ruta para guardar diferentes tipos de reportes (ej. 'rendimiento_keywords')."""
    return get_topic_results_dir(topic) / f"reporte_{report_name}.xlsx"


# --- Funciones de Validación ---

def validate_csv_file(file_path: Path, required_columns: list = None) -> tuple[bool, str]:
    """
    Valida un archivo CSV.
    
    Args:
        file_path: Ruta al archivo CSV
        required_columns: Lista de columnas que deben existir (opcional)
    
    Returns:
        Tupla (es_válido, mensaje)
    """
    try:
        # Verificar que el archivo existe
        if not file_path.exists():
            return False, f"El archivo no existe: {file_path}"
        
        # Verificar extensión
        if file_path.suffix.lower() not in ['.csv']:
            return False, f"El archivo debe ser .csv, encontrado: {file_path.suffix}"
        
        # Intentar leer el archivo
        df = pd.read_csv(file_path, nrows=5)  # Solo leer primeras 5 filas para validar
        
        # Verificar que no esté vacío
        if df.empty:
            return False, "El archivo CSV está vacío"
        
        # Verificar columnas requeridas
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"Faltan columnas requeridas: {missing_columns}"
        
        return True, "Archivo CSV válido"
        
    except Exception as e:
        return False, f"Error al validar archivo CSV: {str(e)}"

def validate_xlsx_file(file_path: Path, required_columns: list = None) -> tuple[bool, str]:
    """
    Valida un archivo Excel.
    
    Args:
        file_path: Ruta al archivo Excel
        required_columns: Lista de columnas que deben existir (opcional)
    
    Returns:
        Tupla (es_válido, mensaje)
    """
    try:
        # Verificar que el archivo existe
        if not file_path.exists():
            return False, f"El archivo no existe: {file_path}"
        
        # Verificar extensión
        if file_path.suffix.lower() not in ['.xlsx', '.xls']:
            return False, f"El archivo debe ser .xlsx o .xls, encontrado: {file_path.suffix}"
        
        # Intentar leer el archivo
        df = pd.read_excel(file_path, nrows=5)  # Solo leer primeras 5 filas para validar
        
        # Verificar que no esté vacío
        if df.empty:
            return False, "El archivo Excel está vacío"
        
        # Verificar columnas requeridas
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"Faltan columnas requeridas: {missing_columns}"
        
        return True, "Archivo Excel válido"
        
    except Exception as e:
        return False, f"Error al validar archivo Excel: {str(e)}"

# --- Estado del Proyecto ---

def crear_dashboard_estado(topic: str) -> dict:
    """
    Crea un dashboard con el estado actual del proyecto.
    
    Args:
        topic: Tema activo
        
    Returns:
        Dict con información del estado del proyecto
    """
    if not topic:
        return {"error": "No hay tema activo"}
    
    dashboard = {
        "tema": topic,
        "pasos_completados": 0,
        "total_pasos": 5,
        "estado_pasos": {},
        "metricas": {},
        "archivos_generados": [],
        "siguiente_accion": ""
    }
    
    # Verificar estado de cada paso
    pasos = [
        {
            "numero": 1,
            "nombre": "Generar Candidatos",
            "archivo": get_human_review_file_path(topic),
            "descripcion": "Búsqueda inicial de contratos candidatos"
        },
        {
            "numero": 2,
            "nombre": "Validación Humana",
            "archivo": get_human_review_file_path(topic),
            "descripcion": "Revisión y validación de contratos por experto"
        },
        {
            "numero": 3,
            "nombre": "Aprender y Refinar",
            "archivo": get_report_path(topic, "rendimiento_keywords"),
            "descripcion": "Análisis de rendimiento y mejora de keywords"
        },
        {
            "numero": 4,
            "nombre": "Entrenar Clasificador",
            "archivo": get_classifier_model_path(topic),
            "descripcion": "Entrenamiento del modelo de clasificación"
        },
        {
            "numero": 5,
            "nombre": "Clasificar Masivamente",
            "archivo": get_predictions_path(topic, 'csv'),
            "descripcion": "Aplicación del modelo a toda la base de datos"
        }
    ]
    
    pasos_completados = 0
    for paso in pasos:
        completado = paso["archivo"].exists()
        if completado:
            pasos_completados += 1
            
        dashboard["estado_pasos"][paso["numero"]] = {
            "nombre": paso["nombre"],
            "completado": completado,
            "descripcion": paso["descripcion"],
            "archivo": paso["archivo"].name if completado else None
        }
    
    dashboard["pasos_completados"] = pasos_completados
    
    # Calcular métricas si están disponibles
    try:
        # Métricas de validación
        review_file = get_human_review_file_path(topic)
        if review_file.exists():
            df_review = pd.read_excel(review_file)
            validation_col = f'Es_{topic.capitalize()}_Validado'
            
            if validation_col in df_review.columns:
                total_contratos = len(df_review)
                validados = len(df_review[df_review[validation_col].isin(['SI', 'NO'])])
                si_validados = len(df_review[df_review[validation_col] == 'SI'])
                
                dashboard["metricas"]["total_contratos"] = total_contratos
                dashboard["metricas"]["validados"] = validados
                dashboard["metricas"]["si_validados"] = si_validados
                dashboard["metricas"]["progreso_validacion"] = (validados / total_contratos) * 100 if total_contratos > 0 else 0
                dashboard["metricas"]["tasa_relevancia"] = (si_validados / validados) * 100 if validados > 0 else 0
        
        # Métricas del modelo
        model_file = get_classifier_model_path(topic)
        if model_file.exists():
            reporte_file = get_report_path(topic, "clasificacion").with_suffix('.json')
            if reporte_file.exists():
                import json
                with open(reporte_file, 'r') as f:
                    reporte = json.load(f)
                    dashboard["metricas"]["precision_modelo"] = reporte.get("SI", {}).get("precision", 0) * 100
                    dashboard["metricas"]["recall_modelo"] = reporte.get("SI", {}).get("recall", 0) * 100
                    dashboard["metricas"]["f1_modelo"] = reporte.get("SI", {}).get("f1-score", 0) * 100
        
        # Métricas de predicciones
        predictions_file = get_predictions_path(topic, 'csv')
        if predictions_file.exists():
            df_pred = pd.read_csv(predictions_file)
            dashboard["metricas"]["total_predicciones"] = len(df_pred)
            if f'Prediccion_{topic.capitalize()}' in df_pred.columns:
                si_predicciones = len(df_pred[df_pred[f'Prediccion_{topic.capitalize()}'] == 'SI'])
                dashboard["metricas"]["si_predicciones"] = si_predicciones
                dashboard["metricas"]["tasa_prediccion"] = (si_predicciones / len(df_pred)) * 100
    
    except Exception as e:
        dashboard["metricas"]["error"] = str(e)
    
    # Determinar siguiente acción
    if pasos_completados == 0:
        dashboard["siguiente_accion"] = "Configura palabras clave y ejecuta el Paso 1"
    elif pasos_completados == 1:
        dashboard["siguiente_accion"] = "Valida los contratos candidatos en el Paso 2"
    elif pasos_completados == 2:
        dashboard["siguiente_accion"] = "Analiza el rendimiento de keywords en el Paso 3"
    elif pasos_completados == 3:
        dashboard["siguiente_accion"] = "Entrena el modelo clasificador en el Paso 4"
    elif pasos_completados == 4:
        dashboard["siguiente_accion"] = "Clasifica toda la base de datos en el Paso 5"
    else:
        dashboard["siguiente_accion"] = "¡Proyecto completado! Revisa los resultados"
    
    return dashboard

def mostrar_dashboard_visual(dashboard: dict):
    """
    Muestra un dashboard visual del estado del proyecto.
    
    Args:
        dashboard: Diccionario con información del estado
    """
    if "error" in dashboard:
        st.error(dashboard["error"])
        return
    
    st.markdown("### 📊 Estado del Proyecto")
    
    # Barra de progreso general
    progreso_general = (dashboard["pasos_completados"] / dashboard["total_pasos"]) * 100
    st.progress(progreso_general / 100)
    st.write(f"**Progreso general:** {dashboard['pasos_completados']}/{dashboard['total_pasos']} pasos completados ({progreso_general:.0f}%)")
    
    # Estado de cada paso
    st.markdown("#### 🔄 Estado de los Pasos")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    cols = [col1, col2, col3, col4, col5]
    for i, col in enumerate(cols, 1):
        paso = dashboard["estado_pasos"][i]
        with col:
            if paso["completado"]:
                st.success(f"✅ Paso {i}")
                st.write(f"**{paso['nombre']}**")
                st.write(f"📁 {paso['archivo']}")
            else:
                st.info(f"⏳ Paso {i}")
                st.write(f"**{paso['nombre']}**")
                st.write("Pendiente")
    
    # Métricas principales
    if dashboard["metricas"]:
        st.markdown("#### 📈 Métricas Principales")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "total_contratos" in dashboard["metricas"]:
                st.metric("📋 Contratos Candidatos", dashboard["metricas"]["total_contratos"])
            if "validados" in dashboard["metricas"]:
                st.metric("✅ Validados", dashboard["metricas"]["validados"])
        
        with col2:
            if "progreso_validacion" in dashboard["metricas"]:
                st.metric("🔄 Progreso Validación", f"{dashboard['metricas']['progreso_validacion']:.1f}%")
            if "tasa_relevancia" in dashboard["metricas"]:
                st.metric("🎯 Tasa Relevancia", f"{dashboard['metricas']['tasa_relevancia']:.1f}%")
        
        with col3:
            if "precision_modelo" in dashboard["metricas"]:
                st.metric("🤖 Precisión Modelo", f"{dashboard['metricas']['precision_modelo']:.1f}%")
            if "total_predicciones" in dashboard["metricas"]:
                st.metric("📊 Total Predicciones", dashboard["metricas"]["total_predicciones"])
    
    # Siguiente acción recomendada
    st.markdown("#### 🎯 Siguiente Acción Recomendada")
    st.info(f"**{dashboard['siguiente_accion']}**")

# --- Wizard de Configuración ---

def mostrar_wizard_configuracion():
    """
    Muestra un asistente paso a paso rediseñado para configurar un nuevo proyecto de clasificación.
    """
    st.markdown("### 🧙‍♂️ Asistente de Configuración Guiado")
    st.markdown("Te guiaré para configurar tu proyecto desde cero.")

    # Inicializar estado del wizard si no existe
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1

    # --- Barra de Progreso ---
    total_steps = 4
    progress = (st.session_state.wizard_step - 1) / (total_steps - 1)
    st.progress(progress)
    st.write(f"**Paso {st.session_state.wizard_step} de {total_steps}**")

    # --- PASO 1: CREAR TEMA ---
    if st.session_state.wizard_step == 1:
        st.markdown("#### 🎯 Paso 1: Define tu Tema de Análisis")
        st.info("El 'tema' es el nombre de tu proyecto (ej: `ciberseguridad`, `obras_civiles`). Todo se guardará bajo este nombre.")
        
        nuevo_tema = st.text_input(
            "Nombre del nuevo tema (en minúsculas, sin espacios):",
            st.session_state.get('wizard_tema', ''),
            placeholder="ejemplo_mi_tema"
        )
        
        if st.button("Siguiente: Configurar Fuentes de Datos ➡️"):
            if nuevo_tema and all(c.islower() or c.isdigit() or c == '_' for c in nuevo_tema):
                st.session_state.wizard_tema = nuevo_tema
                st.session_state.wizard_step = 2
                st.rerun()
            else:
                st.error("❌ El nombre del tema solo puede contener minúsculas, números y guiones bajos (_).")

    # --- PASO 2: CONFIGURAR FUENTES DE DATOS ---
    elif st.session_state.wizard_step == 2:
        st.markdown(f"#### 🗃️ Paso 2: Configura las Fuentes de Datos para `{st.session_state.wizard_tema}`")
        st.info("El sistema intentará cargar datos en este orden: **API → SQL → CSV**. Configura las que necesites.")

        with st.expander("🌐 Configurar API (Opcional)"):
            st.session_state.wizard_api_url = st.text_input("URL Base de la API", st.session_state.get('wizard_api_url', ''))
            st.session_state.wizard_api_key = st.text_input("Token de App (X-App-Token)", st.session_state.get('wizard_api_key', ''), type="password")
            st.session_state.wizard_api_query = st.text_area("Consulta SoQL (Socrata)", st.session_state.get('wizard_api_query', ''), height=100, help="Para APIs tipo Socrata, puedes usar una consulta SoQL.")

        with st.expander("🗄️ Configurar Base de Datos SQL (Opcional)"):
            st.session_state.wizard_sql_db_type = st.selectbox("Tipo de BD", ["postgresql", "mysql", "sqlserver"], key='wizard_sql_db_type')
            st.session_state.wizard_sql_host = st.text_input("Host", st.session_state.get('wizard_sql_host', 'localhost'))
            st.session_state.wizard_sql_port = st.text_input("Puerto", st.session_state.get('wizard_sql_port', '5432'))
            st.session_state.wizard_sql_database = st.text_input("Base de Datos", st.session_state.get('wizard_sql_database', ''))
            st.session_state.wizard_sql_timeout = st.number_input("Timeout de Conexión (segundos)", min_value=5, max_value=120, value=st.session_state.get('wizard_sql_timeout', 10))
            st.session_state.wizard_sql_username = st.text_input("Usuario", st.session_state.get('wizard_sql_username', ''))
            st.session_state.wizard_sql_password = st.text_input("Contraseña", st.session_state.get('wizard_sql_password', ''), type="password")
            st.session_state.wizard_sql_query = st.text_area("Consulta SQL", st.session_state.get('wizard_sql_query', 'SELECT * FROM mi_tabla'))

        st.markdown("---")
        st.markdown("##### 📄 Archivo de Respaldo (Obligatorio)")
        st.warning("Debes subir un archivo CSV o Excel. Se usará si la API y la conexión SQL fallan.")
        st.session_state.wizard_csv_file = st.file_uploader("Sube tu archivo de datos principal", type=['csv', 'xlsx'])

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Paso Anterior"):
                st.session_state.wizard_step = 1
                st.rerun()
        with col2:
            if st.button("Siguiente: Configurar Búsqueda ➡️"):
                if st.session_state.get('wizard_csv_file'):
                    st.session_state.wizard_step = 3
                    st.rerun()
                else:
                    st.error("❌ Debes subir un archivo de respaldo para continuar.")

    # --- PASO 3: CONFIGURAR PARÁMETROS DE BÚSQUEDA ---
    elif st.session_state.wizard_step == 3:
        st.markdown(f"#### 🎨 Paso 3: Personaliza la Búsqueda para `{st.session_state.wizard_tema}`")
        
        # Keywords
        st.markdown("##### 🔑 Palabras Clave Iniciales")
        st.info("Escribe las palabras clave más importantes para tu tema, una por línea.")
        st.session_state.wizard_keywords = st.text_area("Keywords (una por línea)", st.session_state.get('wizard_keywords', ''), height=150)

        # Exclusiones
        st.markdown("##### 🚫 Palabras de Exclusión (Opcional)")
        st.info("Escribe palabras que, si aparecen, descartan un contrato, una por línea.")
        st.session_state.wizard_exclusions = st.text_area("Exclusiones (una por línea)", st.session_state.get('wizard_exclusions', ''), height=100)

        # Columnas de Texto
        st.markdown("##### 📄 Columnas de Texto a Analizar")
        st.info("Selecciona las columnas de tu archivo que contienen texto relevante para el análisis.")
        uploaded_file = st.session_state.get('wizard_csv_file')
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_preview = pd.read_csv(uploaded_file, nrows=0)
                else:
                    df_preview = pd.read_excel(uploaded_file, nrows=0)
                
                available_columns = df_preview.columns.tolist()
                st.session_state.wizard_text_columns = st.multiselect(
                    "Selecciona las columnas a analizar:",
                    options=available_columns,
                    default=st.session_state.get('wizard_text_columns', [])
                )
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Paso Anterior"):
                st.session_state.wizard_step = 2
                st.rerun()
        with col2:
            if st.button("Siguiente: Resumen Final ➡️"):
                if st.session_state.get('wizard_keywords') and st.session_state.get('wizard_text_columns'):
                    st.session_state.wizard_step = 4
                    st.rerun()
                else:
                    st.error("❌ Debes proporcionar al menos una palabra clave y seleccionar al menos una columna de texto.")

    # --- PASO 4: RESUMEN Y FINALIZACIÓN ---
    elif st.session_state.wizard_step == 4:
        st.markdown(f"#### ✅ Paso 4: Resumen y Finalización")
        st.info("Revisa la configuración. Si todo es correcto, finaliza para guardar tu nuevo proyecto.")

        # Mostrar resumen
        st.markdown(f"**Tema del Proyecto:** `{st.session_state.get('wizard_tema')}`")
        st.markdown(f"**Archivo de Respaldo:** `{st.session_state.get('wizard_csv_file').name}`")
        st.markdown(f"**Columnas a Analizar:** `{', '.join(st.session_state.get('wizard_text_columns', []))}`")
        st.markdown(f"**Keywords:** `{len(st.session_state.get('wizard_keywords', '').split())}` palabras clave")
        st.markdown(f"**Exclusiones:** `{len(st.session_state.get('wizard_exclusions', '').split())}` palabras de exclusión")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Paso Anterior"):
                st.session_state.wizard_step = 3
                st.rerun()
        with col2:
            if st.button("🎉 Finalizar y Guardar Configuración", type="primary"):
                try:
                    # Cargar config actual
                    config = load_config()
                    
                    # Guardar archivo CSV
                    tema = st.session_state.wizard_tema
                    uploaded_file = st.session_state.wizard_csv_file
                    csv_path = INPUT_DIR / uploaded_file.name
                    with open(csv_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    # Actualizar config.json
                    config["ACTIVE_TOPIC"] = tema
                    if "TOPICS" not in config: config["TOPICS"] = {}
                    config["TOPICS"][tema] = {
                        "INPUT_FILE_NAME": uploaded_file.name, # Obsoleto pero mantenido por si acaso
                        "TEXT_COLUMNS_TO_COMBINE": st.session_state.wizard_text_columns,
                        "FILTRADO_UNSPSC": {"CODIGOS_DE_INTERES": []} # Se puede añadir en config manual
                    }
                    
                    # Guardar fuentes de datos
                    if "DATA_SOURCE_CONFIG" not in config: config["DATA_SOURCE_CONFIG"] = {}
                    config["DATA_SOURCE_CONFIG"]["CSV"] = {"FILENAME": csv_path.name}
                    config["DATA_SOURCE_CONFIG"]["API"] = {
                        "BASE_URL": st.session_state.get('wizard_api_url', ''),
                        "API_KEY": st.session_state.get('wizard_api_key', ''),
                        "QUERY": st.session_state.get('wizard_api_query', '')
                    }
                    config["DATA_SOURCE_CONFIG"]["SQL"] = {
                        "DB_TYPE": st.session_state.get('wizard_sql_db_type'),
                        "HOST": st.session_state.get('wizard_sql_host'),
                        "PORT": st.session_state.get('wizard_sql_port'),
                        "DATABASE": st.session_state.get('wizard_sql_database'),
                        "CONNECTION_TIMEOUT": st.session_state.get('wizard_sql_timeout', 10),
                        "USERNAME": st.session_state.get('wizard_sql_username'),
                        "PASSWORD": st.session_state.get('wizard_sql_password'),
                        "QUERY": st.session_state.get('wizard_sql_query')
                    }
                    
                    save_config(config)

                    # Guardar keywords
                    keywords_list = [k.strip() for k in st.session_state.wizard_keywords.split('\n') if k.strip()]
                    save_keywords(get_keywords_file_path(tema), {"general": keywords_list})

                    # Guardar exclusiones
                    exclusions_list = [e.strip() for e in st.session_state.wizard_exclusions.split('\n') if e.strip()]
                    if exclusions_list:
                        save_exclusion_words(get_exclusion_file_path(tema), exclusions_list)

                    # Limpiar estado y finalizar
                    for key in list(st.session_state.keys()):
                        if key.startswith('wizard_'):
                            del st.session_state[key]
                    
                    st.success(f"¡Proyecto '{tema}' creado y configurado exitosamente!")
                    st.info("Ahora puedes ir al Panel de Control para empezar a trabajar.")
                    time.sleep(2)
                    st.switch_page("pages/2_Panel_de_Control.py")

                except Exception as e:
                    st.error(f"❌ Ocurrió un error al guardar la configuración: {e}")

def mostrar_boton_wizard():
    """
    Muestra un botón para iniciar el wizard si no hay configuración.
    """
    config = load_config()
    if not config.get("ACTIVE_TOPIC") or not config.get("TOPICS"):
        st.info("🧙‍♂️ **¿Primera vez usando el sistema?** Usa el asistente de configuración para empezar fácilmente.")
        
        if st.button("🚀 Iniciar Asistente de Configuración"):
            st.session_state.mostrar_wizard = True
            st.rerun()
    
    if st.session_state.get('mostrar_wizard', False):
        mostrar_wizard_configuracion()
        
        if st.button("❌ Cerrar Asistente"):
            st.session_state.mostrar_wizard = False
            st.rerun()

# --- Funciones de UI Avanzadas ---

def display_file_explorer(directory: Path, title: str):
    """
    Muestra un explorador de archivos para un directorio específico en Streamlit,
    permitiendo la descarga de cada archivo.

    Args:
        directory (Path): El directorio a explorar.
        title (str): El título a mostrar para la sección.
    """
    st.markdown(f"##### {title}")
    
    # Verificar si el directorio existe y tiene archivos
    if not directory.exists() or not any(f for f in directory.iterdir() if f.is_file()):
        st.info(f"No se encontraron archivos en la carpeta '{directory.name}'.")
        return

    # Listar archivos
    for file_path in sorted(directory.iterdir(), key=lambda f: f.name):
        # Ignorar archivos ocultos (como .DS_Store) y directorios
        if file_path.is_file() and not file_path.name.startswith('.'):
            col1, col2, col3 = st.columns([4, 2, 2])
            with col1:
                st.write(f"📄 **{file_path.name}**")
            with col2:
                try:
                    file_size_kb = file_path.stat().st_size / 1024
                    st.write(f"`{file_size_kb:.2f} KB`")
                except FileNotFoundError:
                    st.write("`-- KB`")
            with col3:
                try:
                    with open(file_path, "rb") as fp:
                        st.download_button(
                            label="Descargar",
                            data=fp,
                            file_name=file_path.name,
                            mime="application/octet-stream", # Mime genérico para descarga
                            key=f"download_{directory.name}_{file_path.name}",
                            use_container_width=True
                        )
                except FileNotFoundError:
                    st.error("No encontrado")

# --- Funciones de Mensajes de Error Amigables ---

def mostrar_error_amigable(error_tipo: str, detalles: str = "", solucion: str = ""):
    """
    Muestra un mensaje de error amigable con solución sugerida.
    
    Args:
        error_tipo: Tipo de error (archivo, configuracion, datos, etc.)
        detalles: Detalles específicos del error
        solucion: Solución sugerida
    """
    errores_amigables = {
        "archivo_no_encontrado": {
            "titulo": "📁 No se encontró el archivo",
            "descripcion": "El archivo que estás buscando no existe en la ubicación esperada.",
            "solucion_default": "Verifica que el archivo esté en la carpeta correcta y que el nombre sea exacto."
        },
        "archivo_corrupto": {
            "titulo": "⚠️ Archivo dañado o con formato incorrecto",
            "descripcion": "El archivo no se puede leer correctamente.",
            "solucion_default": "Intenta abrir el archivo en Excel y guardarlo nuevamente, o usa un archivo diferente."
        },
        "columnas_faltantes": {
            "titulo": "📋 Faltan columnas requeridas",
            "descripcion": "Tu archivo no tiene todas las columnas necesarias para el análisis.",
            "solucion_default": "Revisa la plantilla de ejemplo y asegúrate de que tu archivo tenga las columnas requeridas."
        },
        "configuracion_invalida": {
            "titulo": "⚙️ Configuración incompleta",
            "descripcion": "La configuración del sistema no está completa o tiene errores.",
            "solucion_default": "Ve a la página de inicio y completa la configuración básica."
        },
        "datos_vacios": {
            "titulo": "📊 Sin datos para procesar",
            "descripcion": "No hay datos disponibles para realizar esta operación.",
            "solucion_default": "Asegúrate de haber completado los pasos anteriores correctamente."
        },
        "permisos": {
            "titulo": "🔒 Problema de permisos",
            "descripcion": "No se puede acceder al archivo porque está siendo usado por otra aplicación.",
            "solucion_default": "Cierra Excel u otras aplicaciones que puedan estar usando el archivo."
        },
        "memoria": {
            "titulo": "💾 Archivo demasiado grande",
            "descripcion": "El archivo es muy grande para procesarlo de una vez.",
            "solucion_default": "Divide el archivo en partes más pequeñas o contacta al administrador del sistema."
        }
    }
    
    error_info = errores_amigables.get(error_tipo, {
        "titulo": "❌ Error inesperado",
        "descripcion": "Ocurrió un error que no esperábamos.",
        "solucion_default": "Intenta nuevamente o contacta al soporte técnico."
    })
    
    st.error(f"### {error_info['titulo']}")
    st.write(f"**Qué pasó:** {error_info['descripcion']}")
    if detalles:
        st.write(f"**Detalles:** {detalles}")
    
    solucion_final = solucion or error_info['solucion_default']
    st.info(f"**💡 Solución sugerida:** {solucion_final}")

def validar_archivo_con_mensaje_amigable(file_path: Path, required_columns: list = None) -> bool:
    """
    Valida un archivo y muestra mensajes de error amigables si hay problemas.
    
    Returns:
        bool: True si el archivo es válido, False si hay errores
    """
    try:
        # Verificar que el archivo existe
        if not file_path.exists():
            mostrar_error_amigable("archivo_no_encontrado", 
                                 f"Ruta: {file_path}",
                                 f"Busca el archivo en la carpeta '{file_path.parent.name}' o súbelo nuevamente.")
            return False
        
        # Verificar extensión
        if file_path.suffix.lower() not in ['.csv', '.xlsx', '.xls']:
            mostrar_error_amigable("archivo_corrupto", 
                                 f"Formato encontrado: {file_path.suffix}",
                                 "Usa archivos en formato CSV o Excel (.xlsx, .xls)")
            return False
        
        # Intentar leer el archivo
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, nrows=5)
        else:
            df = pd.read_excel(file_path, nrows=5)
        
        # Verificar que no esté vacío
        if df.empty:
            mostrar_error_amigable("datos_vacios", 
                                 "El archivo no contiene datos",
                                 "Usa un archivo que tenga al menos una fila de datos además de los encabezados.")
            return False
        
        # Verificar columnas requeridas
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                mostrar_error_amigable("columnas_faltantes", 
                                     f"Faltan: {', '.join(missing_columns)}",
                                     f"Agrega estas columnas a tu archivo: {', '.join(missing_columns)}")
                return False
        
        return True
        
    except pd.errors.EmptyDataError:
        mostrar_error_amigable("datos_vacios", 
                             "El archivo está vacío",
                             "Usa un archivo que contenga datos.")
        return False
    except pd.errors.ParserError:
        mostrar_error_amigable("archivo_corrupto", 
                             "El archivo no se puede leer correctamente",
                             "Verifica que el archivo no esté dañado y que tenga el formato correcto.")
        return False
    except PermissionError:
        mostrar_error_amigable("permisos", 
                             f"No se puede acceder a {file_path.name}",
                             f"Cierra {file_path.name} si lo tienes abierto en Excel u otra aplicación.")
        return False
    except MemoryError:
        mostrar_error_amigable("memoria", 
                             f"El archivo {file_path.name} es demasiado grande",
                             "Divide el archivo en partes más pequeñas o contacta al administrador.")
        return False
    except Exception as e:
        mostrar_error_amigable("archivo_corrupto", 
                             f"Error técnico: {str(e)}")
        return False

# --- FUNCIÓN DE EJECUCIÓN MEJORADA ---

def ejecutar_script(script_name: str, show_progress_bar: bool = False, args: list = None):
    """
    Ejecuta un script de Python como un subproceso, mostrando sus logs en vivo.

    Si show_progress_bar es True, muestra una barra de progreso que se actualiza
    parseando la salida de la librería tqdm del script. Si es False, muestra un spinner.

    Args:
        script_name (str): El nombre del archivo .py a ejecutar.
        show_progress_bar (bool): Si es True, muestra una barra de progreso.
        args (list): Lista de argumentos para pasar al script.

    Returns:
        bool: True si la ejecución fue exitosa, False si hubo un error.
    """
    placeholder = st.empty()
    log_expander = st.expander(f"Ver logs de ejecución de '{script_name}'", expanded=False)
    log_output_element = log_expander.code("Iniciando...", language=None)

    progress_bar = None
    if show_progress_bar:
        progress_bar = placeholder.progress(0, text="Iniciando proceso...")
    else:
        placeholder.info(f"⏳ Ejecutando {script_name}, por favor espera...")

    try:
        # Construir comando con argumentos opcionales
        command = [sys.executable, script_name]
        if args:
            command.extend(args)
        
        # Usamos Popen para ejecutar en segundo plano y leer la salida en tiempo real
        proceso = subprocess.Popen(
            command, # Usa el mismo python que corre streamlit
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Combina salida y error en un solo stream
            text=True,
            encoding='utf-8',
            errors='replace', # Añadido para manejar errores de codificación en Windows
            bufsize=1 # Line-buffered
        )

        log_lines = []
        progress_regex = re.compile(r'(\d+)\%\|') # Regex para encontrar "XX%" en la salida de tqdm

        for line in iter(proceso.stdout.readline, ''):
            log_lines.append(line)
            log_output_element.code("".join(log_lines), language=None)

            if show_progress_bar and progress_bar:
                match = progress_regex.search(line)
                if match:
                    # Extraer el porcentaje de la barra de progreso
                    progress_percent = int(match.group(1))

                    # Intentar extraer la descripción que está antes de la barra.
                    # La salida de tqdm usualmente tiene el formato: "Descripción: XX%|...|"
                    line_before_percent = line[:match.start()]
                    description_parts = line_before_percent.rsplit(':', 1)
                    
                    if len(description_parts) > 1:
                        # Si encontramos un ':', usamos el texto que está antes como descripción.
                        description_text = description_parts[0].strip()
                    else:
                        description_text = "Procesando" # Mensaje por defecto si no se encuentra

                    progress_bar.progress(progress_percent / 100, text=f"{description_text}... {progress_percent}%")

        # Esperar a que el proceso termine con un tiempo de espera
        return_code = proceso.wait(timeout=600)  # Timeout de 10 minutos (ajustable)
        
        # Pausa para asegurar que la UI se actualice
        if show_progress_bar:
            time.sleep(1)
            progress_bar.progress(1.0, text="¡Completado!")
            time.sleep(1)

        placeholder.empty()  # Limpiar el placeholder después de la ejecución

        if return_code != 0:
            # El resto del manejo de errores permanece igual
            pass

        if proceso.returncode != 0:
            log_expander.expanded = True # Expandir logs automáticamente en caso de error
            st.error(f"❌ Error al ejecutar el script '{script_name}'. Revisa los logs de arriba para más detalles.")
            return False
        else:            
            return True

    except FileNotFoundError:
        placeholder.error(f"❌ Error Crítico: El script '{script_name}' no fue encontrado.")
        return False
    except Exception as e:
        placeholder.error(f"❌ Ocurrió un error inesperado al intentar ejecutar el script: {e}")
        return False

# --- Funciones de Gestión de Archivos de Configuración (Keywords, Exclusiones) ---

def load_keywords(path_excel):
    """Carga las keywords desde un archivo Excel a un diccionario, limpiándolas."""
    if not Path(path_excel).exists():
        return {}
    try:
        df_keywords = pd.read_excel(path_excel, engine='openpyxl')
        # Convierte cada columna en una entrada de diccionario {nombre_columna: [lista_de_keywords]}
        return {col: [limpiar_termino_busqueda(kw) for kw in df_keywords[col].dropna().astype(str)] for col in df_keywords.columns}
    except Exception:
        return {}

def save_keywords(path_excel, keywords_dict):
    """Guarda un diccionario de keywords en un archivo Excel."""
    # Convierte el diccionario a un DataFrame, rellenando con NaN para que las columnas tengan la misma longitud
    df_to_save = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in keywords_dict.items()]))
    df_to_save.to_excel(path_excel, index=False, engine='openpyxl')

def load_exclusion_words(path_excel):
    """Carga palabras de exclusión desde la primera columna de un archivo Excel."""
    if not Path(path_excel).exists():
        return []
    try:
        df_exclusion = pd.read_excel(path_excel, engine='openpyxl')
        if not df_exclusion.empty:
            # Usa la primera columna, sin importar su nombre
            return df_exclusion.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        return []
    except Exception:
        return []

def save_exclusion_words(path_excel, exclusion_list):
    """Guarda una lista de palabras de exclusión en un archivo Excel."""
    df_to_save = pd.DataFrame({"palabras_de_exclusion": exclusion_list})
    df_to_save.to_excel(path_excel, index=False, engine='openpyxl')


# --- Funciones de Estado y Mantenimiento del Pipeline ---

def check_pipeline_status(topic_name):
    """Verifica el estado de cada paso del pipeline para un tema específico."""
    status = {}
    
    # Paso 1: Generar Candidatos
    review_file = get_human_review_file_path(topic_name)
    if review_file.exists():
        status["Paso 1: Generar Candidatos"] = {"estado": "Completado", "detalle": "Archivo de revisión generado."}
    else:
        status["Paso 1: Generar Candidatos"] = {"estado": "Pendiente", "detalle": "Ejecutar para crear la lista de candidatos."}
        # Si el paso 1 no está, los demás están bloqueados
        for step in ["Paso 2: Validación Humana", "Paso 3: Aprender y Refinar", "Paso 4: Entrenar Clasificador", "Paso 5: Clasificar con Predicciones"]:
            status[step] = {"estado": "Bloqueado", "detalle": "Requiere completar el paso anterior."}
        return status

    # Paso 2: Validación Humana
    df_review = pd.read_excel(review_file, engine='openpyxl')
    validation_col = f'Es_{topic_name.capitalize()}_Validado'
    if validation_col in df_review.columns and df_review[validation_col].notna().any():
        status["Paso 2: Validación Humana"] = {"estado": "Listo para aprender", "detalle": "Hay datos validados."}
    else:
        status["Paso 2: Validación Humana"] = {"estado": "Listo para validar", "detalle": "Valida los contratos en la tabla."}

    # Paso 3: Aprender y Refinar
    model_experto = get_finetuned_model_path(topic_name)
    if model_experto.exists():
        status["Paso 3: Aprender y Refinar"] = {"estado": "Completado", "detalle": "Modelo experto entrenado."}
    else:
        status["Paso 3: Aprender y Refinar"] = {"estado": "Opcional", "detalle": "Puedes entrenar un modelo experto para mejorar la precisión."}

    # Paso 4: Entrenar Clasificador
    model_final = get_classifier_model_path(topic_name)
    if model_final.exists():
        status["Paso 4: Entrenar Clasificador"] = {"estado": "Completado", "detalle": "Clasificador final entrenado."}
    else:
        status["Paso 4: Entrenar Clasificador"] = {"estado": "Listo para entrenar", "detalle": "Entrena el modelo con los datos validados."}

    # Paso 5: Clasificar con Predicciones
    predictions_file = get_predictions_path(topic_name, format='csv')
    if predictions_file.exists():
        status["Paso 5: Clasificar con Predicciones"] = {"estado": "Completado", "detalle": "La clasificación masiva ha sido generada."}
    elif model_final.exists():
        status["Paso 5: Clasificar con Predicciones"] = {"estado": "Listo para predecir", "detalle": "Usa el modelo entrenado para clasificar todo."}
    else:
        status["Paso 5: Clasificar con Predicciones"] = {"estado": "Bloqueado", "detalle": "Requiere un clasificador entrenado."}

    return status

def borrar_resultados_por_tema(topic_name):
    """Borra todos los archivos generados en la carpeta 'resultados' para un tema específico."""
    topic_results_dir = get_topic_results_dir(topic_name)
    if not topic_results_dir.exists():
        return True
    try:
        shutil.rmtree(topic_results_dir)
        return True
    except Exception as e:
        st.error(f"Error al borrar resultados: {e}")
        return False

def display_validation_summary(df, validation_col):
    """Muestra un resumen de las validaciones en la UI."""
    total_rows = len(df)
    validated_rows = df[validation_col].str.strip().isin(['SI', 'NO']).sum()
    si_count = (df[validation_col] == 'SI').sum()
    no_count = (df[validation_col] == 'NO').sum()
    
    st.metric(label="Progreso de Validación", value=f"{validated_rows} / {total_rows}", delta=f"{round((validated_rows/total_rows)*100, 1)}%")
    st.markdown(f"**Detalle:** `{si_count}` contratos marcados como **SI** y `{no_count}` como **NO**.")

# --- Funciones de Monitoreo de Recursos ---
stop_monitoring = threading.Event()

def monitor_resources(interval=5):
    """
    Hilo que monitorea y muestra el uso de CPU, RAM y GPU en intervalos regulares.
    """
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        gpu_available = True
        print("✅ Monitoreo de GPU (NVIDIA) activado.")
    except Exception:
        gpu_available = False
        print("⚠️  Advertencia: No se pudo iniciar el monitoreo de GPU. ¿Tienes una GPU NVIDIA y la librería 'pynvml' instalada?")

    while not stop_monitoring.is_set():
        # Uso de CPU y RAM
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        
        log_line = f"[MONITOR] CPU: {cpu_usage:5.1f}% | RAM: {ram_usage:5.1f}%"

        # Uso de GPU si está disponible
        if gpu_available:
            try:
                gpu_util = nvmlDeviceGetUtilizationRates(handle)
                gpu_mem = nvmlDeviceGetMemoryInfo(handle)
                log_line += f" | GPU Util: {gpu_util.gpu:5.1f}% | GPU VRAM: {(gpu_mem.used / gpu_mem.total) * 100:5.1f}%"
            except Exception:
                log_line += " | GPU: (Error al leer)"

        print(log_line, file=sys.stderr) # Imprimir en stderr para no interferir con la barra de progreso
        time.sleep(interval)