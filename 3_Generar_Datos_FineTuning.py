# --- Dependencias ---
# Este script requiere las siguientes librerías. Puede instalarlas ejecutando en tu terminal:
# pip install pandas

import pandas as pd
from pathlib import Path
import json
import itertools
import random

from utils import get_active_topic, get_active_topic_config, consolidar_validaciones_historicas, get_preprocessed_data_path, get_finetuning_dataset_path

# --- 1. Configuración ---
# Cargar configuración central
TOPIC_NAME = get_active_topic()
topic_config = get_active_topic_config()

if not TOPIC_NAME or not topic_config:
    print("❌ Error: No se pudo cargar la configuración del tema activo.")
    exit()

# Archivo de datos preprocesados (contiene el texto limpio)
PREPROCESSED_DATA_PATH = get_preprocessed_data_path(TOPIC_NAME)

# Archivo de salida para el dataset de fine-tuning
OUTPUT_FINETUNING_DATASET_CSV = get_finetuning_dataset_path(TOPIC_NAME)

# Parámetros de generación
MAX_PAIRS_PER_TYPE = 2000
POSITIVE_SCORE = 1.0
NEGATIVE_SCORE = 0.0
TEXT_COLUMN = 'texto_limpio'
VALIDATION_COLUMN = f'Es_{TOPIC_NAME.capitalize()}_Validado'

# --- 2. Lógica Principal ---

def ejecutar_generacion_dataset():
    print(f"--- Iniciando Generación de Dataset de Fine-Tuning para el Tema: '{TOPIC_NAME.upper()}' ---")

    # --- INICIO DE LA NUEVA LÓGICA ---
    # 1. Cargar la "fuente de la verdad" de las validaciones
    print("Consolidando el historial de todas las validaciones...")
    df_validaciones = consolidar_validaciones_historicas(TOPIC_NAME)

    if df_validaciones.empty:
        print("❌ Error: No se encontraron validaciones en el historial. No se puede generar el dataset.")
        exit()

    # 2. Cargar los datos preprocesados que contienen el texto
    try:
        df_textos = pd.read_parquet(PREPROCESSED_DATA_PATH)
        if 'hash_contrato' not in df_textos.columns or TEXT_COLUMN not in df_textos.columns:
            print(f"❌ Error: El archivo '{PREPROCESSED_DATA_PATH.name}' no contiene las columnas 'hash_contrato' o '{TEXT_COLUMN}'.")
            exit()
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo de datos preprocesados '{PREPROCESSED_DATA_PATH.name}'.")
        exit()

    # 3. Unir las validaciones con los textos usando el hash
    df = pd.merge(df_validaciones, df_textos[['hash_contrato', TEXT_COLUMN]], on='hash_contrato', how='inner')
    # --- FIN DE LA NUEVA LÓGICA ---

    # Separar textos positivos y negativos
    positive_texts = df[df[VALIDATION_COLUMN] == 'SI'][TEXT_COLUMN].tolist()
    negative_texts = df[df[VALIDATION_COLUMN] == 'NO'][TEXT_COLUMN].tolist()

    print(f"Se usarán {len(positive_texts)} contratos positivos ('SI') y {len(negative_texts)} negativos ('NO') para generar los pares.")

    if len(positive_texts) < 2 or len(negative_texts) < 1:
        print("❌ No hay suficientes datos para generar pares. Se necesitan al menos 2 'SI' y 1 'NO'.")
        exit()

    dataset = []

    # --- Generar Pares Positivos ---
    print("Generando pares positivos (SI vs SI)...")
    positive_pairs = list(itertools.combinations(positive_texts, 2))
    random.shuffle(positive_pairs)
    for pair in positive_pairs[:MAX_PAIRS_PER_TYPE]:
        dataset.append({'frase1': pair[0], 'frase2': pair[1], 'score': POSITIVE_SCORE})

    # --- Generar Pares Negativos ---
    print("Generando pares negativos (SI vs NO)...")
    negative_pairs = list(itertools.product(positive_texts, negative_texts))
    random.shuffle(negative_pairs)
    for pair in negative_pairs[:MAX_PAIRS_PER_TYPE]:
        dataset.append({'frase1': pair[0], 'frase2': pair[1], 'score': NEGATIVE_SCORE})

    if not dataset:
        print("❌ No se pudo generar ningún par de entrenamiento.")
        exit()

    # --- Crear y guardar el DataFrame final ---
    df_finetuning = pd.DataFrame(dataset)
    df_finetuning = df_finetuning.drop_duplicates().sample(frac=1).reset_index(drop=True) # Mezclar y eliminar duplicados

    df_finetuning.to_csv(OUTPUT_FINETUNING_DATASET_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ Proceso completado. Dataset de fine-tuning actualizado y guardado en: '{OUTPUT_FINETUNING_DATASET_CSV}'. Total pares: {len(df_finetuning)}")
    print(f"Este archivo ahora puede ser usado por el script '4_Entrenar_Modelo_Preclasificacion.py'.")

if __name__ == "__main__":
    ejecutar_generacion_dataset()