# --- Dependencias ---
import pandas as pd
import joblib
from pathlib import Path
import json
import re
import spacy
from tqdm import tqdm
from utils import (
    cargar_datos_preprocesados, setup_logging, get_active_topic, 
    preparar_texto_para_modelo, get_classifier_model_path, get_human_review_file_path,
    get_predictions_path, get_active_topic_config
)

# --- 1. Configuración ---
# Cargar configuración para obtener el tema activo y las columnas de texto
TOPIC_NAME = get_active_topic()
topic_config = get_active_topic_config()

# Configurar el logger para este script
logger = setup_logging("ejecutar_clasificador", TOPIC_NAME)

if not TOPIC_NAME or not topic_config:
    logger.error(f"Error al cargar la configuración desde 'config.json'.")
    exit()

TEXT_COLUMNS = topic_config.get("TEXT_COLUMNS_TO_COMBINE", ["Objeto del Contrato"])
PREDICTION_THRESHOLD = topic_config.get("PREDICTION_THRESHOLD", 0.80)

# Cargar modelo de spaCy para lematización
try:
    nlp = spacy.load("es_core_news_md", disable=["parser", "ner"])
except OSError:
    logger.error("Modelo de spaCy para español no encontrado. Por favor, ejecute: python -m spacy download es_core_news_md")
    exit()

# Archivos de entrada y salida usando las funciones centralizadas
MODEL_FILENAME = get_classifier_model_path(TOPIC_NAME)
CANDIDATOS_FILENAME = get_human_review_file_path(TOPIC_NAME)
OUTPUT_PREDICTIONS_CSV = get_predictions_path(TOPIC_NAME, format='csv')
OUTPUT_PREDICTIONS_XLSX_SI = get_predictions_path(TOPIC_NAME, format='xlsx')

# Columnas clave
TEXT_FEATURE = 'texto_limpio'
CATEGORICAL_FEATURES = ['Tipo de Contrato', 'Modalidad de Contratacion']
ID_COLUMN = "ID Contrato"
CHUNK_SIZE = topic_config.get("BATCH_SIZE_PREDICTION", 100000)

# --- 2. Lógica de Predicción ---

def ejecutar_prediccion():
    """
    Ejecuta el clasificador entrenado sobre todo el conjunto de datos para generar predicciones masivas.
    """
    logger.info(f"--- Iniciando Script de Predicción para el Tema: '{TOPIC_NAME.upper()}' ---")

    # --- PASO 0: Limpiar resultados anteriores ---
    # Se eliminan los archivos de predicciones anteriores para asegurar que cada ejecución sea nueva.
    logger.info("Limpiando resultados de clasificaciones anteriores para asegurar un inicio limpio...")
    try:
        OUTPUT_PREDICTIONS_CSV.unlink(missing_ok=True)
        OUTPUT_PREDICTIONS_XLSX_SI.unlink(missing_ok=True)
        logger.info("✅ Archivos de predicciones anteriores eliminados (si existían).")
    except Exception as e:
        logger.error(f"Error al intentar eliminar archivos de resultados anteriores: {e}")
        # No es un error fatal, ya que los archivos se sobreescribirán, pero es bueno advertir.
        logger.warning("Continuando con la ejecución. Los archivos existentes serán sobreescritos.")

    # --- PASO 1: Cargar modelo y datos ---
    try:
        model_pipeline = joblib.load(MODEL_FILENAME)
        logger.info(f"✅ Modelo '{MODEL_FILENAME.name}' cargado exitosamente.")
    except FileNotFoundError:
        logger.error(f"❌ Error: No se encontró el modelo '{MODEL_FILENAME.name}'.")
        logger.error("Asegúrate de haber ejecutado el 'Paso 4: Entrenar Clasificador' primero.")
        exit()

    logger.info(f"Cargando datos preprocesados para el tema '{TOPIC_NAME}'...")
    df_full = cargar_datos_preprocesados(TOPIC_NAME)
    if df_full.empty:
        logger.error("❌ Error: No se pudieron cargar los datos preprocesados. Ejecute el script de configuración de fuente.")
        exit()
    
    if ID_COLUMN not in df_full.columns:
        logger.error(f"❌ Error: La columna de identificación '{ID_COLUMN}' no se encuentra en los datos de entrada.")
        exit()

    # Cargar datos de candidatos para obtener subtemas (información contextual)
    try:
        df_candidatos = pd.read_excel(CANDIDATOS_FILENAME, engine='openpyxl')
        subtema_cols = ['Subtemas_Por_Keyword', 'Subtemas_Por_Semantica']
        # Asegurarse de que las columnas existan para evitar errores
        for col in subtema_cols:
            if col not in df_candidatos.columns:
                df_candidatos[col] = ''
        
        # Combinar subtemas en una sola columna
        df_candidatos['Subtema_Detectado'] = df_candidatos[subtema_cols[0]].fillna('') + df_candidatos[subtema_cols[1]].fillna('')
        
        # Unir con el dataframe principal
        df_full = pd.merge(df_full, df_candidatos[[ID_COLUMN, 'Subtema_Detectado']], on=ID_COLUMN, how='left')
        df_full['Subtema_Detectado'] = df_full['Subtema_Detectado'].fillna('No Aplica (No fue candidato inicial)')
    except FileNotFoundError:
        logger.warning(f"⚠️ Advertencia: No se encontró el archivo de candidatos '{CANDIDATOS_FILENAME}'. No se añadirá la columna de subtemas.")
        df_full['Subtema_Detectado'] = 'No disponible'

    # --- Lógica de Procesamiento por Lotes (Baches) ---
    processed_chunks = []
    total_records = len(df_full)
    logger.info(f"Iniciando clasificación en lotes de {CHUNK_SIZE:,} registros. Total a procesar: {total_records:,} registros.")

    for start in tqdm(range(0, total_records, CHUNK_SIZE), desc="Procesando lotes"):
        end = start + CHUNK_SIZE
        df_chunk = df_full.iloc[start:end].copy()

        # --- PASO 2 (por lote): Preparar texto para predicción ---
        # Se utiliza la función centralizada para asegurar consistencia con la fase de candidatos.
        df_chunk = preparar_texto_para_modelo(df_chunk, TEXT_COLUMNS, nlp)

        # Verificar que las columnas categóricas existan antes de usarlas
        features_to_use = [TEXT_FEATURE]
        for cat_col in CATEGORICAL_FEATURES:
            if cat_col in df_chunk.columns:
                features_to_use.append(cat_col)
                df_chunk[cat_col] = df_chunk[cat_col].fillna('Desconocido').astype(str)
            else:
                # Esta advertencia solo se mostrará una vez por ejecución si es necesario
                if start == 0:
                    logger.warning(f"⚠️ Advertencia: La columna categórica '{cat_col}' no se encontró en los datos. No se usará para la predicción.")

        X_chunk = df_chunk[features_to_use]

        # --- PASO 3 (por lote): Realizar Predicciones ---
        probabilidades = model_pipeline.predict_proba(X_chunk)
        
        df_chunk[f'Prediccion_{TOPIC_NAME.capitalize()}'] = ['SI' if p[1] >= PREDICTION_THRESHOLD else 'NO' for p in probabilidades]
        df_chunk[f'Confianza_{TOPIC_NAME.capitalize()}_SI'] = [p[1] for p in probabilidades]

        processed_chunks.append(df_chunk)

    # --- PASO 4: Consolidar y Guardar Resultados ---
    logger.info("Consolidando resultados de todos los lotes...")
    df_final = pd.concat(processed_chunks, ignore_index=True)

    logger.info("Guardando resultados finales...")
    cols_to_drop = ['texto_limpio'] # Las columnas intermedias ya se eliminan en la función central
    output_columns = [col for col in df_final.columns if col not in cols_to_drop]
    df_final_output = df_final[output_columns]

    df_final_output.to_csv(OUTPUT_PREDICTIONS_CSV, index=False, encoding='utf-8-sig')
    logger.info(f"✅ Resultados completos guardados en: '{OUTPUT_PREDICTIONS_CSV.name}' ({len(df_final_output)} filas)")

    # Guardar un archivo Excel solo con los contratos clasificados como 'SI'
    df_filtrado_si = df_final_output[df_final_output[f'Prediccion_{TOPIC_NAME.capitalize()}'] == 'SI'].copy()
    if not df_filtrado_si.empty:
        df_filtrado_si.to_excel(OUTPUT_PREDICTIONS_XLSX_SI, index=False, engine='openpyxl')
        logger.info(f"✅ Contratos relevantes ('SI') guardados en: '{OUTPUT_PREDICTIONS_XLSX_SI.name}' ({len(df_filtrado_si)} filas)")
    else:
        logger.info("ℹ️ No se encontraron contratos clasificados como 'SI'.")

    logger.info(f"\n--- Proceso de Predicción para '{TOPIC_NAME}' completado. ---")

if __name__ == "__main__":
    ejecutar_prediccion()
