# --- Dependencias ---
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import json
import time
import logging
import sys
import torch
import numpy as np
import pickle
import gc
from utils import (
    get_active_topic, get_active_topic_config, get_finetuning_dataset_path, 
    get_finetuned_model_path, get_cache_file_path
)

# --- 1. Configuración del Proceso de Fine-Tuning ---

try:
    config = get_active_topic_config()
    TOPIC_NAME = get_active_topic()
    if not TOPIC_NAME or not config:
        raise ValueError("La clave 'ACTIVE_TOPIC' no está definida o está vacía en config.json")
    
    finetuning_params = config.get("FINETUNING", {})

except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
    print(f"❌ Error al cargar la configuración: {e}")
    exit()

# --- Configuración del logging ---
log_file_path = get_finetuned_model_path(TOPIC_NAME).parent / f"finetuning_{time.strftime('%Y%m%d-%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path, encoding='utf-8'), logging.StreamHandler(sys.stderr)]
)

# --- Configuración Optimizada ---
# --- Modelos Disponibles ---
# Opción 1 (Balanceado, por defecto): 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
# Opción 2 (Máxima Calidad en Español): 'hiiamsid/sentence_similarity_spanish_es'
# Opción 3 (Mejor Calidad Multilingüe): 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
BASE_MODEL_NAME = 'hiiamsid/sentence_similarity_spanish_es'
FINETUNED_MODEL_PATH = get_finetuned_model_path(TOPIC_NAME)

# Parámetros de entrenamiento
NUM_EPOCHS = finetuning_params.get("NUM_EPOCHS", 2)
BATCH_SIZE = finetuning_params.get("BATCH_SIZE", 16)
LEARNING_RATE = finetuning_params.get("LEARNING_RATE", 3e-5)
WARMUP_RATIO = finetuning_params.get("WARMUP_RATIO", 0.1)

INPUT_DATASET_CSV = get_finetuning_dataset_path(TOPIC_NAME)
CACHE_FILE = get_cache_file_path(TOPIC_NAME)

# --- Funciones de Utilidad ---

def cleanup_memory():
    """Libera memoria no utilizada de forma más agresiva."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# --- FUNCIÓN PRINCIPAL OPTIMIZADA ---
def ejecutar_fine_tuning():
    """
    Función principal optimizada para el fine-tuning.
    """
    logging.info(f"--- Iniciando Proceso de Fine-Tuning OPTIMIZADO para: '{TOPIC_NAME.upper()}' ---")

    # --- Verificación de Hardware ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Dispositivo de cómputo: '{device.upper()}'")
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"GPU detectada: {gpu_name}")
    else:
        logging.warning("!! ENTRENAMIENTO CON SOLO CPU SERÁ LENTO !!")

    # --- Fase 1: Preparando datos con CACHÉ ---
    logging.info("\n[Fase 1: Preparando datos de entrenamiento con caché]")
    
    if CACHE_FILE.exists():
        logging.info("📦 Cargando ejemplos desde caché...")
        try:
            with open(CACHE_FILE, 'rb') as f:
                train_examples = pickle.load(f)
            logging.info(f"✅ {len(train_examples)} ejemplos cargados desde caché.")
        except Exception as e:
            logging.warning(f"⚠️ Error al cargar caché: {e}. Recreando dataset...")
            CACHE_FILE.unlink()
            train_examples = None
    else:
        train_examples = None

    if train_examples is None:
        try:
            df_train = pd.read_csv(INPUT_DATASET_CSV)
            logging.info(f"📁 Dataset cargado: {len(df_train)} filas desde '{INPUT_DATASET_CSV.name}'")
        except FileNotFoundError:
            logging.error(f"❌ Error: No se encontró el archivo '{INPUT_DATASET_CSV}'")
            return

        logging.info("🚀 Creando ejemplos de entrenamiento...")
        frases1 = df_train['frase1'].values
        frases2 = df_train['frase2'].values  
        scores = df_train['score'].values.astype(np.float32)
        
        train_examples = [
            InputExample(texts=[f1, f2], label=float(score))
            for f1, f2, score in zip(frases1, frases2, scores)
        ]
        
        logging.info(f"✅ {len(train_examples)} ejemplos creados.")
        
        logging.info("💾 Guardando en caché para futuras ejecuciones...")
        try:
            CACHE_FILE.parent.mkdir(exist_ok=True)
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(train_examples, f)
            logging.info(f"✅ Caché guardado en: {CACHE_FILE}")
        except Exception as e:
            logging.warning(f"⚠️ No se pudo guardar caché: {e}")

    cleanup_memory()

    # --- Fase 2: Carga del Modelo y DataLoader ---
    logging.info("\n[Fase 2: Cargando modelo y configurando DataLoader]")
    
    # --- INICIO DE LA NUEVA LÓGICA DE CARGA ---
    if FINETUNED_MODEL_PATH.exists():
        logging.info(f"🧠 Modelo experto encontrado en '{FINETUNED_MODEL_PATH}'. Cargándolo para re-entrenamiento.")
        model = SentenceTransformer(str(FINETUNED_MODEL_PATH))
    else:
        logging.info(f"📥 No se encontró un modelo experto. Cargando modelo base '{BASE_MODEL_NAME}' desde la web.")
        model = SentenceTransformer(BASE_MODEL_NAME)
    # --- FIN DE LA NUEVA LÓGICA DE CARGA ---

    # [MODIFICACIÓN] Se fuerza num_workers a 0 para eliminar el multiprocesamiento y mejorar la estabilidad.
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    train_loss = losses.CosineSimilarityLoss(model=model)
    
    total_steps = len(train_dataloader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    logging.info(f"📊 Configuración de entrenamiento:")
    logging.info(f"   • Épocas: {NUM_EPOCHS}")
    logging.info(f"   • Batch size: {BATCH_SIZE}")
    logging.info(f"   • Pasos totales: {total_steps}")
    logging.info(f"   • Warmup steps: {warmup_steps}")

    cleanup_memory()

    # --- Fase 3: Ejecución del Fine-Tuning ---
    logging.info(f"\n[Fase 3: Iniciando fine-tuning]")
    
    # Verificación de AMP (Precisión Mixta) para acelerar en GPUs compatibles
    use_amp = False
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 7:
        use_amp = True
        logging.info("✅ AMP (FP16) activado para acelerar entrenamiento en GPU.")

    try:
        start_training = time.time()
        FINETUNED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=NUM_EPOCHS,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
            optimizer_params={'lr': LEARNING_RATE},
            use_amp=use_amp,
            scheduler='warmupcosine'
        )
        
        end_training = time.time()
        logging.info(f"⏱️ Entrenamiento completado en: {(end_training - start_training)/60:.2f} minutos")
        
        logging.info("💾 Guardando modelo final...")
        model.save(str(FINETUNED_MODEL_PATH))
        
        logging.info(f"✅ Modelo guardado en: '{FINETUNED_MODEL_PATH}'")
        
    except Exception as e:
        logging.error(f"❌ Error durante el entrenamiento: {e}")
        return False
        
    finally:
        cleanup_memory()

    logging.info("\n🎉 ¡Fine-tuning optimizado completado exitosamente!")
    return True

if __name__ == "__main__":
    success = ejecutar_fine_tuning()
    if not success:
        sys.exit(1)