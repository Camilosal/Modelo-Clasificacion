# --- Dependencias ---
import pandas as pd
from pathlib import Path
import json
from utils import (
    setup_logging, get_active_topic, get_active_topic_config,
    get_predictions_path, get_active_review_file_path
)

# --- 1. Configuración ---
# Cargar configuración para obtener el tema activo
TOPIC_NAME = get_active_topic()
topic_config = get_active_topic_config()

# Configurar el logger
logger = setup_logging("generar_revision_activa", TOPIC_NAME)

if not TOPIC_NAME or not topic_config:
    logger.error(f"Error al cargar la configuración: No se encontró un tema activo.")
    exit()

# Parámetros para Active Learning
active_learning_config = topic_config.get("ACTIVE_LEARNING", {})
UNCERTAINTY_THRESHOLD_LOW = active_learning_config.get("UNCERTAINTY_THRESHOLD_LOW", 0.80)
UNCERTAINTY_THRESHOLD_HIGH = active_learning_config.get("UNCERTAINTY_THRESHOLD_HIGH", 0.90)
MAX_SAMPLES = active_learning_config.get("MAX_SAMPLES", 250)

# Archivos de entrada y salida usando las funciones centralizadas
PREDICTIONS_CSV = get_predictions_path(TOPIC_NAME, format='csv')
OUTPUT_ACTIVE_REVIEW_XLSX = get_active_review_file_path(TOPIC_NAME)

# Columnas clave
ID_COLUMN = "ID Contrato"
CONFIDENCE_COL = f'Confianza_{TOPIC_NAME.capitalize()}_SI'
PREDICTION_COL = f'Prediccion_{TOPIC_NAME.capitalize()}'
VALIDATION_COL = f'Es_{TOPIC_NAME.capitalize()}_Validado'

# --- 2. Lógica de Selección de Muestras (Active Learning) ---

def generar_lista_revision_activa():
    """
    Selecciona los contratos más inciertos según las predicciones del modelo
    para que un experto los revise.
    """
    logger.info(f"--- Iniciando Generación de Lista de Revisión Activa para el Tema: '{TOPIC_NAME.upper()}' ---")

    # --- PASO 1: Cargar las predicciones ---
    try:
        df_pred = pd.read_csv(PREDICTIONS_CSV)
        logger.info(f"✅ Predicciones cargadas desde '{PREDICTIONS_CSV.name}' ({len(df_pred)} filas).")
    except FileNotFoundError:
        logger.error(f"❌ No se encontró el archivo de predicciones '{PREDICTIONS_CSV.name}'.")
        logger.error("Asegúrate de haber ejecutado el 'Paso 5: Clasificar con Predicciones' primero.")
        exit()

    if CONFIDENCE_COL not in df_pred.columns:
        logger.error(f"❌ La columna de confianza '{CONFIDENCE_COL}' no se encuentra en el archivo de predicciones.")
        exit()

    # --- PASO 2: Identificar las muestras más inciertas ---
    logger.info(f"Buscando contratos con confianza de 'SI' entre {UNCERTAINTY_THRESHOLD_LOW:.0%} y {UNCERTAINTY_THRESHOLD_HIGH:.0%}.")
    uncertain_mask = (df_pred[CONFIDENCE_COL] >= UNCERTAINTY_THRESHOLD_LOW) & (df_pred[CONFIDENCE_COL] <= UNCERTAINTY_THRESHOLD_HIGH)
    df_uncertain = df_pred[uncertain_mask].copy()
    
    logger.info(f"Se encontraron {len(df_uncertain)} contratos en el rango de confianza especificado.")

    # --- PASO 3: Preparar el archivo de salida ---
    # En lugar de ordenar, tomamos una muestra aleatoria para obtener una mezcla representativa
    # de todo el rango, asegurando que se incluyan tanto 'SI' como 'NO' si existen.
    if len(df_uncertain) > MAX_SAMPLES:
        logger.info(f"Seleccionando una muestra aleatoria de {MAX_SAMPLES} contratos de los {len(df_uncertain)} encontrados.")
        df_review_active = df_uncertain.sample(n=MAX_SAMPLES, random_state=42).copy()
    else:
        logger.info(f"Se usarán todos los {len(df_uncertain)} contratos encontrados (límite: {MAX_SAMPLES}).")
        df_review_active = df_uncertain.copy()

    if df_review_active.empty:
        logger.warning("⚠️ No se encontraron contratos en el rango de confianza especificado. Como alternativa, se seleccionarán los más cercanos al 50% de confianza.")
        df_pred['incertidumbre'] = abs(df_pred[CONFIDENCE_COL] - 0.5)
        df_review_active = df_pred.sort_values('incertidumbre', ascending=True).head(MAX_SAMPLES).copy()

    # Añadir la columna de validación vacía que el usuario llenará
    df_review_active[VALIDATION_COL] = ''

    # --- NUEVA LÓGICA: Seleccionar solo columnas clave para la exportación ---
    logger.info("Seleccionando columnas clave para el archivo de revisión activa...")
    
    # Obtener las columnas de texto originales desde la configuración
    text_columns = topic_config.get("TEXT_COLUMNS_TO_COMBINE", [])

    # Definir el conjunto de columnas esenciales para la revisión
    key_columns = [
        ID_COLUMN,
        'hash_contrato',  # Identificador único para cruces
        PREDICTION_COL,   # La predicción del modelo
        VALIDATION_COL,   # La columna para la validación del usuario
    ] + text_columns + [
        CONFIDENCE_COL,   # La confianza del modelo en la predicción 'SI'
        'Subtema_Detectado' # Contexto adicional
    ]

    # Filtrar para mantener solo las columnas que realmente existen en el DataFrame
    existing_key_columns = [col for col in key_columns if col in df_review_active.columns]
    
    # Crear el DataFrame final para exportar y guardarlo
    df_export = df_review_active[existing_key_columns]
    df_export.to_excel(OUTPUT_ACTIVE_REVIEW_XLSX, index=False, engine='openpyxl')
    logger.info(f"✅ Archivo de revisión activa guardado en: '{OUTPUT_ACTIVE_REVIEW_XLSX.name}' con {len(df_export)} contratos y columnas optimizadas.")

if __name__ == "__main__":
    generar_lista_revision_activa()
