#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import shutil
from pathlib import Path
from datetime import datetime

# --- Configuración de Rutas ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "archivos_entrada"
RESULTS_DIR = BASE_DIR / "resultados"
BACKUP_ROOT_DIR = BASE_DIR / "backups"

def get_topic_input_dir(topic: str) -> Path:
    """Devuelve la ruta del directorio de entrada para un tema."""
    return INPUT_DIR / topic

def get_topic_results_dir(topic: str) -> Path:
    """Devuelve la ruta del directorio de resultados para un tema."""
    return RESULTS_DIR / topic

def create_backup(topic_name: str):
    """
    Crea una copia de seguridad completa de la configuración y los modelos para un tema específico.
    """
    print(f"--- Iniciando copia de seguridad para el tema: '{topic_name}' ---")

    # 1. Crear carpeta de destino para el backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT_DIR / f"backup_{topic_name}_{timestamp}"
    
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Carpeta de backup creada en: {backup_dir}")
    except OSError as e:
        print(f"❌ Error Crítico: No se pudo crear la carpeta de backup. {e}")
        return

    # 2. Obtener rutas específicas del tema
    topic_input_dir = get_topic_input_dir(topic_name)
    topic_results_dir = get_topic_results_dir(topic_name)
    
    # 3. Definir los archivos y carpetas a respaldar
    files_to_backup = {
        # Archivo de configuración principal
        "config.json": BASE_DIR / "config.json",
        # Archivos de configuración del tema (nueva estructura)
        "keywords.xlsx": topic_input_dir / "keywords.xlsx",
        "exclusion_words.xlsx": topic_input_dir / "exclusion_words.xlsx",
        # Archivo de datos de entrada del tema
        "entrada_data": topic_input_dir,  # Se copiará toda la carpeta
        # Archivos importantes de resultados
        "contratos_para_revision_humana.xlsx": topic_results_dir / "contratos_para_revision_humana.xlsx",
        "clasificador_v1.joblib": topic_results_dir / "clasificador_v1.joblib",
        "datos_preprocesados.parquet": topic_results_dir / "datos_preprocesados.parquet",
        "finetuning_dataset.csv": topic_results_dir / "finetuning_dataset.csv",
        "historial_entrenamiento.csv": topic_results_dir / "historial_entrenamiento.csv",
        "revision_activa.xlsx": topic_results_dir / "revision_activa.xlsx",
        "reporte_clasificacion.json": topic_results_dir / "reporte_clasificacion.json",
        "reporte_rendimiento_keywords.xlsx": topic_results_dir / "reporte_rendimiento_keywords.xlsx",
        "hashes_validados.csv": topic_results_dir / "hashes_validados.csv"
    }

    dirs_to_backup = {
        # Modelo experto afinado (nueva estructura)
        "modelos_afinados": topic_results_dir / "modelos_afinados",
        # Historial de validaciones
        "historial_validaciones": topic_results_dir / "historial_validaciones",
        # Logs
        "logs": topic_results_dir / "logs"
    }

    # 4. Copiar archivos individuales
    print("\n--- Copiando archivos... ---")
    for dest_name, src_path in files_to_backup.items():
        if dest_name == "entrada_data":  # Caso especial para la carpeta de entrada
            continue  # Se maneja en directorios
        
        if src_path.exists() and src_path.is_file():
            try:
                shutil.copy2(src_path, backup_dir / dest_name)
                print(f"  - ✅ Copiado: {src_path.name}")
            except Exception as e:
                print(f"  - ❌ Error al copiar {src_path.name}: {e}")
        else:
            print(f"  - ⚠️ Omitido (no existe): {src_path.name}")

    # 5. Copiar directorios completos
    print("\n--- Copiando directorios... ---")
    
    # Copiar carpeta de entrada del tema
    entrada_src = topic_input_dir
    if entrada_src.exists() and entrada_src.is_dir():
        try:
            shutil.copytree(entrada_src, backup_dir / f"archivos_entrada_{topic_name}")
            print(f"  - ✅ Copiado directorio: archivos_entrada/{topic_name}")
        except Exception as e:
            print(f"  - ❌ Error al copiar directorio de entrada: {e}")
    
    # Copiar directorios de resultados
    for dest_name, src_path in dirs_to_backup.items():
        if src_path.exists() and src_path.is_dir():
            try:
                shutil.copytree(src_path, backup_dir / dest_name)
                print(f"  - ✅ Copiado directorio: {src_path.name}")
            except Exception as e:
                print(f"  - ❌ Error al copiar el directorio {src_path.name}: {e}")
        else:
            print(f"  - ⚠️ Omitido (no existe): {src_path.name}")

    print("\n🎉 ¡Copia de seguridad completada!")
    print(f"Tus archivos están guardados en: {backup_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Error: Debes proporcionar el nombre del tema como argumento.")
        print("Uso: python create_backup.py <nombre_del_tema>")
        sys.exit(1)
    
    topic_name_arg = sys.argv[1]
    create_backup(topic_name_arg)
