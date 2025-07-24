
import pandas as pd
import re
from pathlib import Path
from collections import Counter
import json
import joblib
from sklearn.metrics import precision_score
from utils import (
    get_active_topic, get_active_topic_config, get_keywords_file_path,
    get_human_review_file_path, get_report_path, load_keywords
)
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 1. Configuración ---
# Cargar configuración central para obtener el tema activo
TOPIC_NAME = get_active_topic()
topic_config = get_active_topic_config()

if not TOPIC_NAME or not topic_config:
    print("❌ Error: No se pudo cargar la configuración del tema activo.")
    exit()

# Archivos de entrada y salida usando las funciones centralizadas de utils.py
INPUT_VALIDATED_XLSX = get_human_review_file_path(TOPIC_NAME)
INPUT_KEYWORDS_EXCEL = get_keywords_file_path(TOPIC_NAME)
OUTPUT_FEEDBACK_REPORT_XLSX = get_report_path(TOPIC_NAME, 'rendimiento_keywords')

# Columnas y parámetros clave
TEXT_COLUMN_CLEAN = 'texto_limpio'
VALIDATION_COLUMN = f'Es_{TOPIC_NAME.capitalize()}_Validado'
ID_COLUMN = "ID Contrato"
NUM_SUGGESTIONS = 20
MIN_WORD_LENGTH = 4 # Aumentado a 4 para evitar palabras muy cortas y poco significativas

# --- 2. Lógica Principal ---

def ejecutar_analisis_feedback():
    print(f"--- Iniciando Análisis de Feedback para el Tema: '{TOPIC_NAME.upper()}' ---")

    # --- PASO 1: Cargar datos validados ---
    try:
        df = pd.read_excel(INPUT_VALIDATED_XLSX, engine='openpyxl')
        df_validated = df[df[VALIDATION_COLUMN].str.strip().isin(['SI', 'NO'])].copy()
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo validado '{INPUT_VALIDATED_XLSX}'.")
        print("Asegúrate de haber completado el Paso 2 (Validación Humana) primero.")
        exit()

    if df_validated.empty:
        print("ℹ️ No se encontraron contratos validados. El reporte no puede ser generado.")
        exit()

    # --- PASO 2: Analizar rendimiento de keywords y temas existentes ---
    print("\nAnalizando rendimiento de keywords y temas existentes...")
    
    # Cargar keywords existentes
    try:
        from utils import load_keywords
        keywords_dict = load_keywords(INPUT_KEYWORDS_EXCEL)
    except Exception as e:
        print(f"⚠️  Advertencia: No se pudieron cargar las keywords existentes. Error: {e}")
        keywords_dict = {}

    # 2.1: Calcular rendimiento por keyword
    keyword_data = []
    for tema, keywords in keywords_dict.items():
        for keyword in keywords:
            # Detectar si la keyword está presente en el texto limpio
            keyword_detected = df_validated[TEXT_COLUMN_CLEAN].apply(lambda text: bool(re.search(r'\b' + re.escape(keyword.lower()) + r'\b', str(text).lower())))

            # Comparar con la validación humana
            true_positives = ((df_validated[VALIDATION_COLUMN] == 'SI') & keyword_detected).sum()
            false_positives = ((df_validated[VALIDATION_COLUMN] == 'NO') & keyword_detected).sum()
            total_detections = keyword_detected.sum()

            # Calcular precisión solo si hubo detecciones
            precision = true_positives / total_detections if total_detections > 0 else 0.0  
            
            keyword_data.append({
                'Keyword': keyword,
                'Tema': tema,
                'Aciertos': true_positives,
                'Fallos': false_positives,
                'Total': total_detections,
                'Precision': precision
            })
            
    keyword_feedback = pd.DataFrame(keyword_data)

    # 2.2: Calcular rendimiento por tema (agregado)
    theme_data = []
    for tema in keywords_dict.keys():
        # Filtrar las filas donde al menos una keyword del tema fue detectada
        tema_keywords = keywords_dict[tema]
        tema_detected = df_validated[TEXT_COLUMN_CLEAN].apply(lambda text: any(re.search(r'\b' + re.escape(keyword.lower()) + r'\b', str(text).lower()) for keyword in tema_keywords))
        df_tema = df_validated[tema_detected]
        
        # Calcular precisión si hubo detecciones, sino, dejar como NaN
        if not df_tema.empty:
            y_true = df_tema[VALIDATION_COLUMN].apply(lambda x: 1 if x == 'SI' else 0)  # Convertir validaciones a numérico
            y_pred = [1] * len(df_tema)  # Asumimos que todas las detecciones del tema son positivas (para calcular precisión)
            precision = precision_score(y_true, y_pred)  # Calcular precisión para el tema
        else:
            precision = 0.0  # No hubo detecciones, precisión 0
            
        theme_data.append({
            'Tema': tema,
            'Aciertos': (df_tema[VALIDATION_COLUMN] == 'SI').sum(),  # Contar 'SI' en los detectados por el tema
            'Fallos': (df_tema[VALIDATION_COLUMN] == 'NO').sum(),    # Contar 'NO' en los detectados
            'Total': len(df_tema),  # Total de documentos donde se detectó al menos una keyword del tema
            'Precision': precision
        })

    theme_feedback = pd.DataFrame(theme_data)
    semantic_feedback = pd.DataFrame() # Para futura implementación
    sugerencias_df = pd.DataFrame()

    # --- PASO 3: Generar sugerencias de nuevas keywords ---
    # Identificar contratos 'SI' que NO fueron detectados por keywords
    df_positivos_sin_kw = df_validated[
        (df_validated[VALIDATION_COLUMN] == 'SI') &
        (df_validated['Metodo_Deteccion'] == 'SEMANTICO')
    ].copy()

    if not df_positivos_sin_kw.empty:
        print(f"\nSe analizarán {len(df_positivos_sin_kw)} contratos 'SI' no detectados por keywords para encontrar nuevas sugerencias.")

        # --- 3.1: Inicializar filtros y TF-IDF ---
        # Lista de Stop words (personalízala según tu necesidad)
        stop_words = set([
            "de", "la", "el", "en", "y", "a", "los", "con", "para", "que", "un", "una", "por", "o", "es", "se", "del", "al", "lo", "como", "mas", "su", "sus", "este", "esta", "entre", "e", "u", "les", "si", "sin", "ser", "ha", "son", "muy", "ya", "asi", "mismo", "vez", "solo", "tambien", "sobre", "todo", "todos", "cada", "uno", "unos", "unas", "otro", "otra", "otros", "otras", "pero", "porque", "cuando", "donde", "cual", "cuales", "quien", "quienes", "ese", "esa", "esos", "esas", "esto", "aqui", "ahi", "alli", "aca", "mucho", "muchos", "muchas", "poco", "pocos", "pocas", "tanto", "tan", "misma", "mismas", "mismos", "durante", "mediante", "objeto", "contrato", "prestacion", "servicio", "servicios"
        ])

        # Cargar keywords existentes para no sugerirlas de nuevo
        keywords_existentes = set()
        try:
            from utils import load_keywords
            keywords_data = load_keywords(INPUT_KEYWORDS_EXCEL)
            for kws in keywords_data.values():
                # Las keywords ya vienen limpias y en minúsculas desde load_keywords
                keywords_existentes.update(kws)
            print(f"Se cargaron {len(keywords_existentes)} keywords existentes para exclusión.")
        except Exception as e:
            print(f"⚠️  Advertencia: No se pudieron cargar las keywords existentes. Error: {e}")

        # --- 3.2: Extracción de n-gramas y cálculo de métricas ---
        print("Extrayendo n-gramas, calculando frecuencias y TF-IDF...")
        
        vectorizer = TfidfVectorizer(
            stop_words=list(stop_words),
            ngram_range=(1, 3), # Unigramas, bigramas y trigramas
            min_df=2,           # Debe aparecer en al menos 2 contratos
            max_df=0.95,        # No más del 95% de los contratos
            max_features=5000
        )
        
        tfidf_matrix = vectorizer.fit_transform(df_positivos_sin_kw[TEXT_COLUMN_CLEAN])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray().mean(axis=0)
        frequency_per_ngram = tfidf_matrix.astype(bool).sum(axis=0).A1
        ngram_data = list(zip(feature_names, frequency_per_ngram, tfidf_scores))
        
        # Ajustar puntuaciones para priorizar n-gramas más largos
        adjusted_ngram_data = []
        for ngram, frequency, score in ngram_data:
            n = len(ngram.split())
            if n == 2: score *= 1.5  # Prioridad a bigramas
            elif n == 3: score *= 2.0  # Máxima prioridad a trigramas
            adjusted_ngram_data.append((ngram, frequency, score))
        
        # --- 3.3: Filtrar y preparar sugerencias finales ---
        print("Filtrando y preparando sugerencias...")
        sugerencias = []
        for ngram, frequency, score in adjusted_ngram_data:
            if ngram in keywords_existentes or len(ngram) < MIN_WORD_LENGTH:
                continue
            if len(ngram.split()) > 1 and any(len(w) < MIN_WORD_LENGTH for w in ngram.split()):
                continue
            sugerencias.append((ngram, frequency, score))
            
        sugerencias.sort(key=lambda x: x[2], reverse=True)
        sugerencias = sugerencias[:NUM_SUGGESTIONS]
        
        if sugerencias:
            print(f"\n✅ Se generaron {len(sugerencias)} sugerencias de keywords:")
            sugerencias_df = pd.DataFrame(sugerencias, columns=['Keyword Sugerida', 'Apariciones', 'TF-IDF'])
            for _, row in sugerencias_df.iterrows():
                print(f"   - '{row['Keyword Sugerida']}' (Apariciones: {row['Apariciones']}, TF-IDF: {row['TF-IDF']:.4f})")
        else:
            print("ℹ️ No se encontraron nuevas sugerencias de keywords después de aplicar filtros.")
    else:
        print("ℹ️ No hay textos positivos sin keywords para analizar. No se generarán sugerencias en este ciclo.")

    # --- PASO 4: Guardar el reporte en un archivo Excel ---
    print(f"\nGuardando reporte en '{OUTPUT_FEEDBACK_REPORT_XLSX}'...")
    with pd.ExcelWriter(OUTPUT_FEEDBACK_REPORT_XLSX, engine='openpyxl') as writer:
        if not keyword_feedback.empty:
            keyword_feedback.to_excel(writer, sheet_name='Rendimiento por Keyword', index=False)
        if not theme_feedback.empty:
            theme_feedback.to_excel(writer, sheet_name='Rendimiento por Tema', index=False)
        if not sugerencias_df.empty:
            sugerencias_df.to_excel(writer, sheet_name='Sugerencias Keywords', index=False)

    print("\n🎉 ¡Proceso de retroalimentación completado!")
    print(f"Abre el archivo Excel '{OUTPUT_FEEDBACK_REPORT_XLSX}' para ver las nuevas sugerencias.")

if __name__ == "__main__":
    ejecutar_analisis_feedback()
