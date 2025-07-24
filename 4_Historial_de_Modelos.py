import streamlit as st
import pandas as pd
from pathlib import Path
from utils import get_active_topic, load_config

st.set_page_config(
    page_title="Historial de Modelos",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Historial de Rendimiento de Modelos")
st.markdown("Aquí puedes ver la evolución del rendimiento de tus modelos de clasificación a lo largo del tiempo. Cada fila representa un entrenamiento.")

# --- 1. Cargar todos los historiales disponibles ---
RESULTS_DIR = Path(__file__).resolve().parent.parent / "resultados"
# [CORRECCIÓN] Buscar recursivamente en las carpetas de cada tema
history_files = list(RESULTS_DIR.glob("**/reporte_historial_entrenamiento.csv"))

if not history_files:
    st.info("Aún no se ha entrenado ningún modelo. El historial aparecerá aquí después de que completes el Paso 4 por primera vez.")
else:
    # Cargar y concatenar todos los archivos de historial
    df_list = []
    for file in history_files:
        try:
            df_list.append(pd.read_csv(file))
        except Exception as e:
            st.warning(f"No se pudo leer el archivo de historial '{file.name}': {e}")
    
    if not df_list:
        st.error("Se encontraron archivos de historial, pero no se pudieron cargar. Pueden estar vacíos o corruptos.")
        st.stop()

    df_history = pd.concat(df_list, ignore_index=True)
    
    # Convertir timestamp a datetime para ordenar y graficar
    df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
    df_history = df_history.sort_values(by='timestamp', ascending=False).reset_index(drop=True)

    # --- 2. Filtros y Visualización ---
    st.markdown("---")
    
    # Obtener lista de todos los temas en el historial
    all_topics = df_history['topic'].unique().tolist()
    
    # Filtro por tema
    selected_topics = st.multiselect(
        "Filtrar por Tema:",
        options=all_topics,
        default=all_topics
    )
    
    if not selected_topics:
        st.warning("Por favor, selecciona al menos un tema para ver los resultados.")
        st.stop()

    df_filtered = df_history[df_history['topic'].isin(selected_topics)]

    # --- 3. Mostrar Tabla de Historial ---
    st.markdown("### Historial de Entrenamientos")
    
    # Formatear columnas para mejor visualización
    columns_to_format_percent = [
        'accuracy', 'si_precision', 'si_recall', 'si_f1-score',
        'no_precision', 'no_recall', 'no_f1-score'
    ]
    format_dict = {col: "{:.2%}" for col in columns_to_format_percent}
    
    st.dataframe(
        df_filtered.style.format(format_dict),
        use_container_width=True
    )

    # --- 4. Gráficos de Evolución ---
    st.markdown("---")
    st.markdown("### Evolución del Rendimiento")
    
    if len(df_filtered) > 1:
        # Preparar datos para el gráfico
        df_chart = df_filtered.sort_values(by='timestamp', ascending=True)
        
        # Seleccionar métricas a graficar
        metrics_to_plot = st.multiselect(
            "Selecciona las métricas a visualizar en el gráfico:",
            options=['accuracy', 'si_precision', 'si_recall', 'si_f1-score'],
            default=['accuracy', 'si_f1-score']
        )
        
        if metrics_to_plot:
            # Configurar el gráfico
            st.line_chart(
                df_chart,
                x='timestamp',
                y=metrics_to_plot,
                color='topic' # Usar el tema para diferenciar líneas si hay varios seleccionados
            )
        else:
            st.info("Selecciona al menos una métrica para visualizar el gráfico de evolución.")
            
    else:
        st.info("Se necesita más de un entrenamiento para mostrar la evolución del rendimiento.")
