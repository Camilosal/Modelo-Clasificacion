# --- Dependencias ---
# Este script requiere las siguientes librerías. Puedes instalarlas ejecutando en tu terminal:
# pip install pandas joblib scikit-learn seaborn matplotlib

import pandas as pd
import joblib
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC # type: ignore
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from utils import (
    setup_logging, get_active_topic, get_active_topic_config,
    get_human_review_file_path, get_classifier_model_path, get_report_path
)

# --- 1. Configuración ---
# Cargar configuración central
TOPIC_NAME = get_active_topic()
topic_config = get_active_topic_config()

# Configurar el logger para este script
logger = setup_logging("entrenamiento_clasificador", TOPIC_NAME)

if not TOPIC_NAME or not topic_config:
    logger.error("Error al cargar la configuración desde 'config.json'.")
    exit()

# Archivos de entrada y salida usando las funciones centralizadas
INPUT_VALIDATED_XLSX = get_human_review_file_path(TOPIC_NAME)
OUTPUT_MODEL_FILENAME = get_classifier_model_path(TOPIC_NAME)
OUTPUT_METRICS_FILENAME = get_report_path(TOPIC_NAME, 'clasificacion.json')
OUTPUT_CONFUSION_MATRIX_IMG = get_report_path(TOPIC_NAME, 'matriz_confusion.png')
HISTORY_FILE = get_report_path(TOPIC_NAME, 'historial_entrenamiento.csv')

# Modelo a utilizar, cargado desde la configuración
CLASSIFIER_MODEL = topic_config.get("CLASSIFIER_MODEL", "RandomForestClassifier")

# Columnas clave para el modelo
TARGET_COLUMN = f'Es_{TOPIC_NAME.capitalize()}_Validado'
TEXT_FEATURE = 'texto_limpio'
CATEGORICAL_FEATURES = ['Tipo de Contrato', 'Modalidad de Contratacion'] 

# --- 2. Lógica Principal de Entrenamiento ---

def ejecutar_entrenamiento_final():
    logger.info(f"--- Iniciando Pipeline de Entrenamiento para el Tema: '{TOPIC_NAME.upper()}' ---")
    start_time = time.time()

    # --- PASO 1: Cargar y Preparar los Datos Validados ---
    try:
        df = pd.read_excel(INPUT_VALIDATED_XLSX, dtype=str, engine='openpyxl')
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo validado '{INPUT_VALIDATED_XLSX}' para el tema '{TOPIC_NAME}'.")
        logger.error("Asegúrate de haber ejecutado los scripts anteriores y de haber guardado tus validaciones en el Paso 2.")
        exit()

    # Filtrar solo las filas que han sido validadas y limpiar
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna('') # Asegurarse de que la columna de texto no tenga NaNs
    df = df[df[TARGET_COLUMN].str.strip().isin(['SI', 'NO'])]

    if len(df) < 10: # Chequeo de sanidad
        logger.error(f"No hay suficientes datos validados ('SI'/'NO') en la columna '{TARGET_COLUMN}' para entrenar un modelo.")
        exit()

    logger.info(f"Se usarán {len(df)} contratos validados de acuerdo con la columna '{TARGET_COLUMN}' para el entrenamiento.")

    # Verificar y preparar columnas categóricas
    valid_categorical_features = []
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna('Desconocido').astype(str)
            valid_categorical_features.append(col)
        else:
            logger.warning(f"La columna categórica '{col}' no se encontró en los datos y será ignorada.")

    # Preparar características (X) y objetivo (y)
    X = df[[TEXT_FEATURE] + valid_categorical_features]
    y = df[TARGET_COLUMN].map({'SI': 1, 'NO': 0})

    # --- PASO 2: Dividir Datos para Entrenamiento y Evaluación ---
    # `stratify=y` es CRUCIAL para asegurar que la proporción de SI/NO sea la misma
    # en los conjuntos de entrenamiento y prueba, especialmente si hay desbalance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    logger.info(f"Datos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba.")
    logger.info("10%|---------|") # Initial progress


    # --- PASO 3: Construir y Entrenar el Pipeline de Clasificación ---
    # El pipeline se encarga de procesar los datos y alimentar al modelo de forma consistente.
    text_processor = TfidfVectorizer(stop_words=None, max_features=5000, ngram_range=(1, 2))
    categorical_processor = OneHotEncoder(handle_unknown='ignore')

    # Procesador para características categóricas.
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_processor, TEXT_FEATURE),
            ('categorical', categorical_processor, valid_categorical_features)
        ],
        remainder='drop'
    )

    # --- Selección dinámica del clasificador ---
    logger.info(f"Seleccionando el modelo: {CLASSIFIER_MODEL}")
    classifiers = {
        # [OPTIMIZACIÓN] n_jobs=-1 utiliza todos los núcleos de CPU disponibles para acelerar el entrenamiento y la predicción.
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
        "LogisticRegression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000, n_jobs=-1),
        "SVC": SVC(random_state=42, class_weight='balanced', probability=True) # probability=True es VITAL para predict_proba
    }

    classifier_instance = classifiers.get(CLASSIFIER_MODEL)
    if classifier_instance is None:
        logger.warning(f"Modelo '{CLASSIFIER_MODEL}' no reconocido. Usando RandomForestClassifier por defecto.")
        classifier_instance = classifiers["RandomForestClassifier"]


    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier_instance)
    ])
    # `class_weight='balanced'` ayuda al modelo a prestar más atención a la clase minoritaria (probablemente 'SI')

    logger.info("40%|---------|") # Approximated progress after preprocessor setup
    logger.info("Entrenando el modelo con los datos validados...")
    model_pipeline.fit(X_train, y_train)
    logger.info("✅ Modelo entrenado exitosamente.")
    logger.info("90%|---------|") # Approximated progress after training

    end_time = time.time()
    training_time = end_time - start_time

    # --- PASO 4: Evaluar el Rendimiento del Modelo ---
    logger.info("Evaluando el modelo en el conjunto de prueba (datos que nunca ha visto)...")
    y_pred = model_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    logger.info(f"🎯 Accuracy (Precisión Global): {accuracy:.4f}")
    logger.info(f"⚖️ Matthews Correlation Coefficient (MCC): {mcc:.4f} (1 es perfecto)")

    logger.info("📋 Reporte de Clasificación Detallado:")
    target_names = [f'NO (No {TOPIC_NAME.capitalize()})', f'SI ({TOPIC_NAME.capitalize()})']
    logger.info("\n" + classification_report(y_test, y_pred, target_names=target_names))

    # Generar reporte y matriz de confusión para guardarlos
    report_dict = classification_report(y_test, y_pred, target_names=['NO', 'SI'], output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # --- PASO 5: Guardar Métricas y Modelo ---
    logger.info("Guardando métricas de rendimiento del modelo...")
    # Combinar el reporte de clasificación con otras métricas para un solo archivo JSON
    report_dict["accuracy"] = accuracy
    report_dict["mcc"] = mcc
    report_dict["confusion_matrix"] = conf_matrix.tolist()

    with open(OUTPUT_METRICS_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Métricas guardadas en '{OUTPUT_METRICS_FILENAME}'. Tiempo total de entrenamiento: {training_time:.2f} segundos.")

    # Guardamos todo el pipeline, que incluye el vectorizador de texto.
    # Esto es crucial para poder clasificar nuevos datos de la misma manera.
    joblib.dump(model_pipeline, OUTPUT_MODEL_FILENAME)
    logger.info(f"✅ Modelo (pipeline completo) guardado en '{OUTPUT_MODEL_FILENAME}'.")

    # --- PASO 6: Guardar Visualización de Matriz de Confusión ---
    logger.info("Generando y guardando visualización de la matriz de confusión...")
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                    xticklabels=['NO', 'SI'], yticklabels=['NO', 'SI'])
        plt.xlabel('Predicción del Modelo')
        plt.ylabel('Etiqueta Real')
        plt.title(f'Matriz de Confusión - {TOPIC_NAME.capitalize()}')

        plt.savefig(OUTPUT_CONFUSION_MATRIX_IMG, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Matriz de confusión guardada como imagen en '{OUTPUT_CONFUSION_MATRIX_IMG}'.")
    except Exception as e:
        logger.error(f"No se pudo generar o guardar la imagen de la matriz de confusión: {e}")

    # --- Guardar métricas en el historial ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    si_metrics = report_dict['SI']
    no_metrics = report_dict['NO']
    accuracy = report_dict['accuracy']
    mcc = report_dict['mcc']

    new_entry = {
        'timestamp': timestamp,
        'topic': TOPIC_NAME,
        'accuracy': accuracy,
        'mcc': mcc,
        'si_precision': si_metrics['precision'],
        'si_recall': si_metrics['recall'],
        'si_f1-score': si_metrics['f1-score'],
        'no_precision': no_metrics['precision'],
        'no_recall': no_metrics['recall'],
        'no_f1-score': no_metrics['f1-score'],
        'support_si': si_metrics['support'],
        'support_no': no_metrics['support']
    }

    df_history = pd.DataFrame([new_entry])

    if HISTORY_FILE.exists():
        df_history.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    else:
        df_history.to_csv(HISTORY_FILE, mode='w', header=True, index=False)
    logger.info(f"Métricas guardadas en el historial: '{HISTORY_FILE.name}'")

    logger.info("🎉 ¡Entrenamiento completado!")

if __name__ == "__main__":
    ejecutar_entrenamiento_final()
    print("100%|██████████|") # Final progress