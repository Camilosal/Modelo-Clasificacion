# --- Dependencias ---
import pandas as pd
from pathlib import Path
import json
from utils import (
    setup_logging, get_active_topic, get_human_review_file_path, 
    get_active_review_file_path
)

# --- 1. Configuración ---
# Cargar configuración para obtener el tema activo
TOPIC_NAME = get_active_topic()

# Configurar el logger
logger = setup_logging("consolidar_validaciones", TOPIC_NAME)

if not TOPIC_NAME:
    logger.error("Error al cargar la configuración: No se encontró un tema activo.")
    exit()

# Archivos de entrada y salida usando las funciones centralizadas
MAIN_REVIEW_XLSX = get_human_review_file_path(TOPIC_NAME)
ACTIVE_REVIEW_XLSX = get_active_review_file_path(TOPIC_NAME)

# Columnas clave
ID_COLUMN = "ID Contrato"
VALIDATION_COL = f'Es_{TOPIC_NAME.capitalize()}_Validado'

# --- 2. Lógica de Consolidación ---

def consolidar_validaciones():
    """
    Combina las validaciones del ciclo principal y del ciclo activo en un
    único archivo de entrenamiento, dando prioridad a las validaciones del ciclo activo.
    """
    logger.info(f"--- Iniciando Consolidación de Validaciones para el Tema: '{TOPIC_NAME.upper()}' ---")

    # --- PASO 1: Cargar ambos archivos de revisión ---
    try:
        df_main = pd.read_excel(MAIN_REVIEW_XLSX, engine='openpyxl')
        logger.info(f"✅ Archivo de revisión principal cargado: {len(df_main)} filas.")
    except FileNotFoundError:
        logger.error(f"❌ No se encontró el archivo de revisión principal '{MAIN_REVIEW_XLSX.name}'. No se puede consolidar.")
        exit()

    try:
        df_active = pd.read_excel(ACTIVE_REVIEW_XLSX, engine='openpyxl')
        logger.info(f"✅ Archivo de revisión activa cargado: {len(df_active)} filas.")
    except FileNotFoundError:
        logger.warning(f"⚠️ No se encontró el archivo de revisión activa. Se procederá solo con el archivo principal.")
        df_active = pd.DataFrame()

    # --- PASO 2: Filtrar solo las filas validadas (con chequeos de robustez) ---
    if VALIDATION_COL not in df_main.columns:
        logger.error(f"❌ La columna de validación '{VALIDATION_COL}' no se encuentra en el archivo principal '{MAIN_REVIEW_XLSX.name}'.")
        logger.error("Asegúrate de que el archivo no haya sido modificado manualmente. No se puede continuar.")
        exit()
    df_main_validated = df_main[df_main[VALIDATION_COL].isin(['SI', 'NO'])].copy()

    df_active_validated = pd.DataFrame()
    if not df_active.empty:
        if VALIDATION_COL in df_active.columns:
            df_active_validated = df_active[df_active[VALIDATION_COL].isin(['SI', 'NO'])].copy()
        else:
            logger.warning(f"⚠️ La columna de validación '{VALIDATION_COL}' no se encontró en el archivo de revisión activa. Se ignorarán sus datos.")

    logger.info(f"Validaciones encontradas en archivo principal: {len(df_main_validated)}")
    if not df_active_validated.empty:
        logger.info(f"Validaciones encontradas en archivo de revisión activa: {len(df_active_validated)}")

    # Obtener IDs antes de la consolidación para rastrear cambios
    main_validated_ids = set(df_main_validated[ID_COLUMN])
    active_validated_ids = set(df_active_validated[ID_COLUMN]) if not df_active.empty else set()

    # --- PASO 3: Consolidar y dar prioridad a la revisión activa ---
    df_consolidated = pd.concat([df_main_validated, df_active_validated], ignore_index=True)
    
    total_before_dedup = len(df_consolidated)
    df_consolidated.drop_duplicates(subset=[ID_COLUMN], keep='last', inplace=True)
    total_after_dedup = len(df_consolidated)

    num_overwritten = total_before_dedup - total_after_dedup
    if num_overwritten > 0:
        logger.info(f"{num_overwritten} validaciones del archivo principal fueron sobreescritas por las de la revisión activa (prioridad más alta).")

    logger.info(f"Total de validaciones únicas consolidadas: {len(df_consolidated)}.")

    # --- PASO 4: Actualizar el archivo de entrenamiento principal (Lógica corregida) ---
    # La lógica anterior con `update` no añadía nuevos contratos de la revisión activa.
    # Esta nueva lógica asegura que tanto las actualizaciones como las nuevas adiciones se manejen correctamente.

    # 1. Identificar todos los IDs de contratos que tienen una validación definitiva.
    validated_ids = df_consolidated[ID_COLUMN].tolist()

    # 2. Crear una base de datos principal que excluya cualquier versión anterior de los contratos validados.
    df_main_sin_validados = df_main[~df_main[ID_COLUMN].isin(validated_ids)]

    # 3. Concatenar la base (sin los validados) con el conjunto de datos consolidado y definitivo.
    #    Esto añade tanto los contratos actualizados como los completamente nuevos de la revisión activa.
    df_final = pd.concat([df_main_sin_validados, df_consolidated], ignore_index=True)

    df_final.to_excel(MAIN_REVIEW_XLSX, index=False, engine='openpyxl')
    logger.info(f"✅ Archivo '{MAIN_REVIEW_XLSX.name}' actualizado con {len(df_final)} filas en total.")
    logger.info(f"Se procesaron {len(df_consolidated)} validaciones únicas.")

if __name__ == "__main__":
    consolidar_validaciones()
