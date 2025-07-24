# --- Dependencias ---
# Este script requiere las siguientes librerías. Es necesario instalarlas ejecutando en la terminal en el entorno virtual o de trabajo:
# pip install "numpy<2" pandas openpyxl torch sentence-transformers spacy
# Y luego descargar el modelo de español: python -m spacy download es_core_news_lg

import pandas as pd
import re
import time
from pathlib import Path
import torch # type: ignore
from sentence_transformers import SentenceTransformer, util
import json
import spacy
import logging
from utils import (
    get_active_topic, get_active_topic_config, load_config,
    get_preprocessed_data_path, preparar_texto_para_modelo,
    get_keywords_file_path, get_exclusion_file_path, get_human_review_file_path, limpiar_termino_busqueda,
    get_finetuned_model_path, get_topic_logs_dir, get_cache_file_path, cargar_datos_preprocesados
)

# --- 1. Configuración Global ---
BASE_DIR = Path(__file__).resolve().parent
import threading
import time
import sys, utils
import hashlib

# La configuración se carga desde un archivo central 'config.json'.
TOPIC_NAME = get_active_topic()
topic_config = get_active_topic_config()
config = load_config() # Cargar la configuración completa

if not TOPIC_NAME or not topic_config:
    # Si no hay tema, no podemos continuar.
    # Usamos print en lugar de logging porque el logger aún no está configurado.
    print("ERROR: No se pudo cargar la configuración del tema activo. Asegúrate de que esté bien configurado en config.json.")
    exit()

# --- Configuración del logging ---
# Ahora que tenemos el tema, podemos crear la carpeta de logs específica.
LOGS_DIR = get_topic_logs_dir(TOPIC_NAME)
log_file_path = LOGS_DIR / f"seleccion_candidatos_{time.strftime('%Y%m%d-%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ])

logging.info(f"Tema activo configurado: {TOPIC_NAME}")

# Cargar las columnas de texto a combinar desde la configuración del tema
TEXT_COLUMNS = topic_config.get("TEXT_COLUMNS_TO_COMBINE", ["Objeto del Contrato", "objeto_del_contrato", "Objeto Contrato"])

# Cargar el modelo de spaCy para lematización
try:
    nlp = spacy.load("es_core_news_md", disable=["parser", "ner"])  # type: ignore
except OSError:
    logging.error("Modelo de spaCy para español no encontrado. Por favor, ejecute: python -m spacy download es_core_news_md")
    exit()

# Archivos de entrada y salida usando las nuevas funciones de utils
INPUT_KEYWORDS_EXCEL = get_keywords_file_path(TOPIC_NAME)
INPUT_EXCLUSION_WORDS_EXCEL = get_exclusion_file_path(TOPIC_NAME)
OUTPUT_FINAL_REVIEW_XLSX = get_human_review_file_path(TOPIC_NAME)

# Parámetros de filtrado y texto
ID_COLUMN = "ID Contrato" # Columna de ID única para identificar contratos

# Parámetros del modelo semántico
FINETUNED_MODEL_PATH = get_finetuned_model_path(TOPIC_NAME)

if FINETUNED_MODEL_PATH.exists():
    MODEL_NAME = str(FINETUNED_MODEL_PATH)
    logging.info(f"Usando modelo afinado localmente para '{TOPIC_NAME}'.")
else:
    MODEL_NAME = 'hiiamsid/sentence_similarity_spanish_es'
    logging.warning(f"No se encontró un modelo afinado localmente en '{FINETUNED_MODEL_PATH}'. Se usará un modelo genérico: '{MODEL_NAME}'.")

# Umbral de similitud desde configuración o valor por defecto
SIMILARITY_THRESHOLD = config.get("TOPICS", {}).get(TOPIC_NAME, {}).get("SIMILARITY_THRESHOLD", 0.7)

# [OPTIMIZACIÓN] Tamaño del lote para procesar los datos. Reduce el uso de RAM.
CHUNK_SIZE = 100000

# --- 2. Funciones Auxiliares ---

def cargar_keywords_desde_excel(path_excel):
    try:
        df_keywords = pd.read_excel(path_excel, engine='openpyxl')
        keywords_por_tema = {col: [limpiar_termino_busqueda(kw) for kw in df_keywords[col].dropna().astype(str)] for col in df_keywords.columns}
        logging.info(f"Palabras clave cargadas y limpiadas desde '{path_excel}'. {len(keywords_por_tema)} temas encontrados.")
        return keywords_por_tema
    except FileNotFoundError:
        logging.error(f"El archivo de keywords '{path_excel}' no fue encontrado. Asegúrate de que exista para el tema '{TOPIC_NAME}' definido en config.json.")
        exit()
    
def cargar_palabras_exclusion(path_excel):
    """
    Carga una lista de palabras de exclusión desde la primera columna de un archivo Excel.
    Este archivo es opcional.
    """
    try:
        df_exclusion = pd.read_excel(path_excel, engine='openpyxl')
        if not df_exclusion.empty:
            # Usar la primera columna, sea cual sea su nombre
            exclusion_list = df_exclusion.iloc[:, 0].dropna().astype(str).str.lower().tolist()
            logging.info(f"{len(exclusion_list)} palabras de exclusión cargadas desde '{path_excel}'.")
            return exclusion_list
        return []
    except FileNotFoundError:
        logging.info(f"Archivo de palabras de exclusión '{path_excel}' no encontrado. No se aplicará este filtro.")
        return []
    except Exception as e:
        logging.warning(f"No se pudo leer el archivo de exclusión '{path_excel}': {e}")
        return []

def buscar_terminos(texto, keywords_por_tema):
    temas, kws = set(), set()
    for tema, keywords in keywords_por_tema.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', texto):
                temas.add(tema)
                kws.add(kw)
    return ", ".join(sorted(list(temas))), ", ".join(sorted(list(kws)))  # Devolución más clara

def contiene_palabra_exclusion(texto, exclusion_list):
    """Verifica si el texto contiene alguna de las palabras de la lista de exclusión."""
    # Añadir esta línea para manejar datos no textuales y evitar el TypeError
    if not isinstance(texto, str): return False
    for word in exclusion_list:
        if re.search(r'\b' + re.escape(word) + r'\b', texto):
            return True # Retorno explícito para mayor claridad
    return False

# --- Fases del Pipeline ---

def _fase_2_limpiar_y_excluir(df: pd.DataFrame) -> pd.DataFrame:
    """Combina columnas de texto, limpia el texto y aplica filtro de exclusión."""
    print("\n[Fase 2: Limpieza de texto y filtro por exclusión]")
    start_time = time.time()
    
    # Se utiliza la función centralizada para asegurar consistencia con la fase de clasificación.
    logging.info("Iniciando preparación de texto (combinación, limpieza y lematización)...")
    df = preparar_texto_para_modelo(df, TEXT_COLUMNS, nlp)
    logging.info("Texto preparado exitosamente.")

    exclusion_list = cargar_palabras_exclusion(INPUT_EXCLUSION_WORDS_EXCEL)
    if exclusion_list:
        initial_count = len(df)
        exclusion_mask = df['texto_limpio'].apply(lambda texto: contiene_palabra_exclusion(texto, exclusion_list))
        df = df[~exclusion_mask].copy()
        num_excluidos = initial_count - len(df)
        if num_excluidos > 0:
            logging.info(f"Se excluyeron {num_excluidos} contratos. Quedan {len(df)} para el siguiente análisis.")

    end_time = time.time()
    logging.info(f"Fase 2 completada. Tiempo: {end_time - start_time:.2f} segundos.")
    return df

def _fase_3_buscar_por_keywords(df: pd.DataFrame, keywords_dict: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Busca por keywords y separa el DataFrame en candidatos y restantes."""
    print("\n[Fase 3: Buscando candidatos por palabras clave]")
    start_time = time.time()

    resultados_kw = df['texto_limpio'].apply(lambda t: buscar_terminos(t, keywords_dict))
    df[['Subtemas_Por_Keyword', 'Keywords_Detectados']] = pd.DataFrame(resultados_kw.tolist(), index=df.index)

    df_keyword_candidates = df[df['Keywords_Detectados'] != ''].copy()
    df_keyword_candidates['Metodo_Deteccion'] = 'KEYWORD'
    logging.info(f"Se encontraron {len(df_keyword_candidates)} candidatos por keywords.")

    df_restantes = df[df['Keywords_Detectados'] == ''].copy()
    end_time = time.time()
    logging.info(f"Fase 3 completada. Tiempo: {end_time - start_time:.2f} segundos.")
    return df_keyword_candidates, df_restantes

def _fase_4_buscar_por_semantica(df_restantes: pd.DataFrame, keywords_dict: dict, cached_embeddings: dict) -> pd.DataFrame:
    """Analiza semánticamente los contratos restantes para encontrar más candidatos."""
    print("\n[Fase 4: Buscando candidatos por similitud semántica]")
    start_time = time.time()

    if df_restantes.empty:
        logging.info("Todos los contratos filtrados fueron encontrados por keywords. No se requiere análisis semántico.")
        end_time = time.time()
        logging.info(f"Fase 4 (análisis semántico) omitida. Tiempo: {end_time - start_time:.2f} segundos.")
        return pd.DataFrame()    

    # --- INICIO DE MODIFICACIÓN: Lógica de Caché de Embeddings ---
    logging.info(f"Analizando semánticamente {len(df_restantes)} contratos restantes (usando caché)...")
    
    # Identificar qué contratos de este lote ya tienen embeddings en el caché
    hashes_en_lote = df_restantes['hash_contrato'].tolist()
    hashes_a_codificar = [h for h in hashes_en_lote if h not in cached_embeddings]
    
    if hashes_a_codificar:
        logging.info(f"Se generarán embeddings para {len(hashes_a_codificar)} nuevos contratos en este lote.")
        textos_a_codificar = df_restantes[df_restantes['hash_contrato'].isin(hashes_a_codificar)]['texto_limpio'].tolist()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(MODEL_NAME, device=device)
        
        nuevos_embeddings = model.encode(
            textos_a_codificar, 
            convert_to_tensor=True, 
            show_progress_bar=True,
            batch_size=32
        )
        
        # Actualizar el diccionario de caché en memoria
        for hash_val, embedding in zip(hashes_a_codificar, nuevos_embeddings):
            cached_embeddings[hash_val] = embedding.cpu() # Guardar en CPU para evitar problemas de memoria
    else:
        logging.info("Todos los embeddings para este lote ya estaban en caché. Saltando generación.")

    # Ensamblar el tensor de embeddings para el lote completo desde el caché
    contract_embeddings = torch.stack([cached_embeddings[h] for h in hashes_en_lote])
    # --- FIN DE MODIFICACIÓN ---

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    theme_descriptions = [f"Contratos o servicios sobre {t.replace('_', ' ').lower()} relacionados con {TOPIC_NAME}: {', '.join(kws)}" for t, kws in keywords_dict.items()]

    logging.info("Generando embeddings para temas...")
    theme_embeddings = model.encode(theme_descriptions, convert_to_tensor=True, show_progress_bar=True)
    # Los embeddings de contratos ya están cargados o generados arriba, no necesitamos regenerarlos

    semantic_hits = util.semantic_search(contract_embeddings, theme_embeddings, top_k=5)

    semantic_candidates_data = []
    for i, hits in enumerate(semantic_hits):
        if not hits: continue
        valid_hits = [h for h in hits if h['score'] >= SIMILARITY_THRESHOLD]
        if valid_hits:
            valid_hits.sort(key=lambda x: x['score'], reverse=True)
            matched_subthemes = [list(keywords_dict.keys())[h['corpus_id']] for h in valid_hits]

            original_index = df_restantes.index[i]
            candidate_info = df_restantes.loc[original_index].to_dict()
            candidate_info.update({
                'Metodo_Deteccion': 'SEMANTICO',
                'Subtemas_Por_Semantica': ", ".join(matched_subthemes),
                'Similitud_Semantica_Max': round(valid_hits[0]['score'], 4)
            })
            semantic_candidates_data.append(candidate_info)

    if not semantic_candidates_data:
        logging.info("No se encontraron nuevos candidatos semánticos que superen el umbral.")
        end_time = time.time()
        logging.info(f"Fase 4 completada. Tiempo: {end_time - start_time:.2f} segundos.")
        return pd.DataFrame()
    
    df_semantic_candidates = pd.DataFrame(semantic_candidates_data)
    logging.info(f"Se encontraron {len(df_semantic_candidates)} nuevos candidatos por análisis semántico.")
    end_time = time.time()
    logging.info(f"Fase 4 completada. Tiempo: {end_time - start_time:.2f} segundos.")
    return df_semantic_candidates

def _fase_final_preparar_archivo(df_final: pd.DataFrame):
    """Prepara y guarda el DataFrame final con columnas clave en un archivo Excel para revisión humana."""
    logging.info("\n[Fase Final: Generando archivo para revisión humana con columnas clave]")
    start_time = time.time()
    
    validation_col = f'Es_{TOPIC_NAME.capitalize()}_Validado'
    df_final[validation_col] = ''

    # Rellenar valores nulos en columnas de contexto para evitar problemas
    df_final.fillna({
        'Similitud_Semantica_Max': 0.0, 
        'Subtemas_Por_Semantica': '', 
        'Subtemas_Por_Keyword': '', 
        'Keywords_Detectados': ''
    }, inplace=True)

    # --- Incorporar validaciones previas si existen ---
    # Esta lógica ahora usa 'hash_contrato' que es más robusto que 'ID Contrato
    if OUTPUT_FINAL_REVIEW_XLSX.exists():
        logging.info(f"Archivo de revisión anterior encontrado. Intentando preservar validaciones existentes...")
        try:
            df_previous = pd.read_excel(OUTPUT_FINAL_REVIEW_XLSX, engine='openpyxl')
            validation_col = f'Es_{TOPIC_NAME.capitalize()}_Validado'
            
            # Verificar si la columna de validación existe y tiene datos válidos
            if validation_col in df_previous.columns and df_previous[validation_col].notna().any():
                # Crear un mapa de ID -> Validación para una búsqueda rápida y eficiente
                # Solo se incluyen las filas que ya tienen una validación ('SI' o 'NO')
                previous_validations = df_previous[df_previous[validation_col].isin(['SI', 'NO'])]
                validation_map = pd.Series(previous_validations[validation_col].values, index=previous_validations[ID_COLUMN]).to_dict()
                
                # Aplicar el mapa a la columna de validación del nuevo DataFrame.
                # Esto llenará 'SI' o 'NO' donde los IDs coincidan, y dejará el resto como estaba (vacío).
                df_final[validation_col] = df_final[ID_COLUMN].map(validation_map).fillna('')
                
                num_preserved = df_final[df_final[validation_col] != ''].shape[0]
                if num_preserved > 0:
                    logging.info(f"✅ Se preservaron {num_preserved} validaciones de la ejecución anterior.")
            else:
                logging.info("El archivo de revisión anterior no contiene validaciones. Se generará un archivo nuevo.")
        except Exception as e:
            logging.warning(f"No se pudieron cargar o procesar las validaciones previas: {e}")

    # --- INICIO DE MODIFICACIÓN: Reordenar columnas para mejor visualización ---
    # Columnas que queremos al principio del archivo Excel para facilitar la revisión.
    # Usamos TEXT_COLUMNS que se carga desde la configuración global.
    columnas_clave_inicio = [
        'hash_contrato', # ¡MUY IMPORTANTE! Para el seguimiento de validaciones
        ID_COLUMN,
        'familia_unspsc', # La columna UNSPSC que se quiere añadir
    ] + TEXT_COLUMNS + [
        validation_col,
        'Metodo_Deteccion',
        'Similitud_Semantica_Max',
        'Keywords_Detectados',
        'Subtemas_Por_Keyword',
        'Subtemas_Por_Semantica'
    ]

    # Filtrar para mantener solo las columnas que realmente existen en el DataFrame
    columnas_clave_existentes = [col for col in columnas_clave_inicio if col in df_final.columns]    
    # Seleccionar únicamente las columnas clave para el archivo de salida, haciéndolo más ligero.
    df_final_ordenado = df_final[columnas_clave_existentes]

    df_final_ordenado.to_excel(OUTPUT_FINAL_REVIEW_XLSX, index=False, engine='openpyxl')
    # --- FIN DE MODIFICACIÓN ---

    end_time = time.time()
    logging.info(f"Proceso completado. Archivo de revisión guardado en: '{OUTPUT_FINAL_REVIEW_XLSX}'")
    logging.info(f"Total de candidatos a revisar: {len(df_final)}. Tiempo: {end_time - start_time:.2f} segundos.")

# --- 3. Lógica Principal del Pipeline ---

def ejecutar_preprocesamiento():
    start_time = time.time()
    logging.info(f"--- Iniciando Pipeline de Preprocesamiento para el Tema: '{TOPIC_NAME.upper()}' ---")

    # --- FASE 1: Cargar datos preprocesados ---
    # El archivo de datos preprocesados se genera en la página "Fuente de Datos".
    # Este script consume ese archivo estandarizado.
    logging.info("\n[Fase 1: Cargando datos preprocesados]")
    df_filtrado = cargar_datos_preprocesados(TOPIC_NAME)
    if df_filtrado.empty:
        logging.error(f"No se pudieron cargar los datos preprocesados para el tema '{TOPIC_NAME}'.")
        logging.error("Asegúrate de haber generado el archivo de datos estandarizado desde la página 'Fuente de Datos' primero.")
        exit()

    # --- NUEVO: Aplicar filtro UNSPSC en la selección de candidatos ---
    logging.info("\n[Fase 1.5: Aplicando filtro por códigos UNSPSC]")
    unspsc_config = topic_config.get("FILTRADO_UNSPSC", {})
    codigos_interes = unspsc_config.get("CODIGOS_DE_INTERES", [])
    columna_unspsc = unspsc_config.get("COLUMNA_UNSPSC", "") # Leer la columna desde la config

    if codigos_interes and columna_unspsc and columna_unspsc in df_filtrado.columns:
        initial_rows = len(df_filtrado)
        logging.info(f"Aplicando filtro por {len(codigos_interes)} códigos UNSPSC de interés en la columna '{columna_unspsc}'.")
        
        # --- INICIO DE LA MODIFICACIÓN: Extracción y limpieza de códigos UNSPSC ---
        # Log de diagnóstico para ver los datos originales
        logging.info(f"Primeros 5 valores (no nulos) de '{columna_unspsc}' antes de la extracción: {df_filtrado[columna_unspsc].dropna().head().tolist()}")

        # 1. Crear una columna temporal con el código UNSPSC completo y limpio.
        df_filtrado['codigo_completo_limpio'] = df_filtrado[columna_unspsc].astype(str).str.strip()

        # 2. Extraer la parte numérica del código.
        #    - Elimina el prefijo 'V1.' o 'v1.' si existe.
        #    - Se queda con la parte entera si hay un decimal (ej. '81111500.0' -> '81111500').
        #    - Elimina cualquier caracter no numérico restante.
        df_filtrado['codigo_completo_limpio'] = df_filtrado['codigo_completo_limpio'].str.replace(r'^[Vv]1\.?\s*', '', regex=True)
        df_filtrado['codigo_completo_limpio'] = df_filtrado['codigo_completo_limpio'].str.split('.').str[0]
        df_filtrado['codigo_completo_limpio'] = df_filtrado['codigo_completo_limpio'].str.replace(r'\D', '', regex=True)

        # 3. Extraer explícitamente la familia (los primeros 4 dígitos) para el filtro.
        df_filtrado['familia_unspsc'] = df_filtrado['codigo_completo_limpio'].str[:4]

        # Log de diagnóstico para verificar la extracción de la familia
        logging.info(f"Primeros 5 valores de la familia UNSPSC extraída ('familia_unspsc'): {df_filtrado['familia_unspsc'].dropna().head().tolist()}")

        # 4. Aplicar el filtro comparando la columna de familia con los códigos de interés.
        codigos_interes_str = [str(c) for c in codigos_interes]
        mask = df_filtrado['familia_unspsc'].isin(codigos_interes_str)
        df_filtrado = df_filtrado[mask].copy()

        # 5. Eliminar las columnas temporales
        df_filtrado.drop(columns=['codigo_completo_limpio', 'familia_unspsc'], inplace=True)
        # --- FIN DE LA MODIFICACIÓN ---

        logging.info(f"Filtrado UNSPSC completado. Registros antes: {initial_rows:,}, después: {len(df_filtrado):,}.")
    
    elif codigos_interes:
        logging.warning(f"Se configuraron códigos UNSPSC pero la columna '{columna_unspsc}' no se encontró o no está configurada. El filtro no se aplicará.")
    else:
        logging.info("No se han configurado códigos UNSPSC de interés. Se procesarán todos los datos para la selección de candidatos.")
    # --- FIN DEL NUEVO BLOQUE ---

    # --- INICIO DE MODIFICACIÓN: Optimización de rendimiento ---
    # Se realiza la limpieza de texto y exclusión UNA SOLA VEZ sobre todo el dataset
    # para aprovechar al máximo el multiprocesamiento de spaCy (n_process=-1).
    logging.info("\n[Fase 2: Optimizando rendimiento con pre-procesamiento masivo de texto...]")
    df_limpio_completo = _fase_2_limpiar_y_excluir(df_filtrado)
    # --- FIN DE MODIFICACIÓN ---
    
    # --- INICIO DE MODIFICACIÓN: Cargar caché de embeddings ---
    import joblib
    cache_file = get_cache_file_path(TOPIC_NAME, "embeddings")
    if cache_file.exists():
        logging.info(f"Cargando caché de embeddings desde {cache_file.name}...")
        cached_embeddings = joblib.load(cache_file)
    else:
        logging.info("No se encontró caché de embeddings. Se creará uno nuevo.")
        cached_embeddings = {}
    # --- FIN DE MODIFICACIÓN ---

    try:
        keywords_dict = cargar_keywords_desde_excel(INPUT_KEYWORDS_EXCEL)
        if not keywords_dict: exit()

        # --- INICIO DE MODIFICACIÓN: Búsqueda por keywords sobre el dataset ya limpio ---
        df_keyword_candidates, df_restantes = _fase_3_buscar_por_keywords(df_limpio_completo, keywords_dict)
        # --- FIN DE MODIFICACIÓN ---

        # La búsqueda semántica, que consume más memoria, se mantiene en lotes.
        all_semantic_candidates = []
        total_records = len(df_restantes)
        logging.info(f"Iniciando procesamiento en lotes de {CHUNK_SIZE:,} registros. Total a procesar: {total_records:,} registros.")

        for start in range(0, total_records, CHUNK_SIZE):
            end = start + CHUNK_SIZE
            logging.info(f"\n--- Procesando lote: {start+1} a {min(end, total_records)} de {total_records} ---")
            df_chunk = df_restantes.iloc[start:end].copy()
            df_semantic_candidates_chunk = _fase_4_buscar_por_semantica(df_chunk, keywords_dict, cached_embeddings)
            if not df_semantic_candidates_chunk.empty:
                all_semantic_candidates.append(df_semantic_candidates_chunk)

        # Consolidar todos los candidatos encontrados
        logging.info("\nConsolidando resultados de todos los lotes...")
        df_final = pd.concat([df_keyword_candidates] + all_semantic_candidates, ignore_index=True)
        
        if not df_final.empty:
            _fase_final_preparar_archivo(df_final)
        else:
            logging.warning("No se encontraron candidatos en ningún lote. No se generará archivo de revisión.")
    finally:
        # --- INICIO DE MODIFICACIÓN: Guardar caché de embeddings ---
        joblib.dump(cached_embeddings, cache_file)
        logging.info(f"Caché de embeddings ({len(cached_embeddings)} elementos) actualizado y guardado en {cache_file.name}.")
        # --- FIN DE MODIFICACIÓN ---
        # Calcular y mostrar el tiempo total de ejecución
        end_time = time.time()
        duration = end_time - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        print(f"\n--- Tiempo total de ejecución: {minutes} minutos y {seconds} segundos. ---")

if __name__ == "__main__":
    logging.info("Script de preprocesamiento iniciado.")
    ejecutar_preprocesamiento()
