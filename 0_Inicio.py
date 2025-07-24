# --- Dependencias ---
import streamlit as st
import pandas as pd
from pathlib import Path
import io
from datetime import datetime
import time

# --- Importar funciones de utilidad ---
from utils import (
    load_config, get_active_topic, crear_dashboard_estado,
    INPUT_DIR,
    check_pipeline_status, mostrar_boton_wizard
)

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Sistema de Clasificación de Contratos Temas de Interés",
    page_icon="assets/logo.png",
    layout="wide"
)

# --- Logo de la Entidad ---
st.sidebar.image("assets/logo.png")

# --- Sección "Acerca de" en la barra lateral ---
st.sidebar.markdown("### Acerca del Sistema")
st.sidebar.info(
    """
    **Versión:** 1.0.1\n
    **Desarrollado por:** Manuel Camilo Saldarriaga - GAEC/EMAE\n
    **Contacto:** camilosal@me.com\n
    """
)
st.sidebar.markdown(
    "Este sistema utiliza IA para ayudar en la clasificación de contratos, "
    "aprendiendo del conocimiento experto para mejorar continuamente."
)

# --- Cargar configuración ---
config = load_config()
ACTIVE_TOPIC = config.get("ACTIVE_TOPIC", "")

st.title("🏠 Sistema de Clasificación de Contratos CCE")

pipeline_status = check_pipeline_status(ACTIVE_TOPIC)
st.markdown("## 🔎 ¿Qué es este sistema?")
st.markdown(""" 
### 🚀 Más Allá de la Búsqueda: Un Asistente Inteligente a tu Lado
Este sistema va más allá de un buscador de terminos y palabras. Es una herramienta que **analiza, comprende y aprende** de tu pericia para transformar la manera en que encuentras y clasificas la información.

- **🧠 Inteligencia Adaptativa:** Olvídate de las reglas fijas. El sistema aprende directamente de tus decisiones como experto en el tema.
- **🎯 Precisión Evolutiva:** Cada contrato que validas (`SI` o `NO`) lo hace más certero, refinando su capacidad de clasificación con cada interacción.
- **🤝 Tu Conocimiento, Escalado:** Convierte tu experiencia y criterio en un potente motor de clasificación que trabaja de forma automática y masiva.
- **📈 Mejora Continua:** A medida que interactúas, el sistema se vuelve más inteligente, adaptándose a tus necesidades y preferencias.
- **🔄 Clasificar no solo objetos de contratos:** puedes personalizarlo para clasificar cualquier campo de texto o categorico que configures
""")
with st.container():
    st.markdown("---")
    st.markdown("#### 🎯 ¿Cómo funciona?")
    st.markdown("""
    1. **Tú configuras** las palabras clave y parámetros iniciales
    2. **El sistema busca** contratos candidatos usando múltiples métodos
    3. **Tú validas** los resultados (marcas como relevantes o no)
    4. **Generas un modelo** de clasificación inteligente basado en tu retroalimentación
    5. **Realizas análisis** automáticos de nuevos contratos con el modelo creado 
    """)

# --- Selección de Flujo de Trabajo ---
st.markdown("---")
st.markdown("## ✍️ ¿Qué quieres hacer?")

# Cargar todos los temas disponibles
all_topics = list(config.get("TOPICS", {}).keys())

# Determinar el índice del tema activo
try:
    active_topic_index = all_topics.index(ACTIVE_TOPIC) if ACTIVE_TOPIC in all_topics else 0
except ValueError:
    active_topic_index = 0

# Crear una lista de opciones para el selectbox
# Usar un identificador único para la opción de crear nuevo tema
CREATE_NEW_TOPIC_OPTION = "__CREAR_NUEVO_TEMA__"
topic_options = all_topics + [CREATE_NEW_TOPIC_OPTION]

# Formatear las opciones para mostrarlas al usuario
def format_topic_option(option):
    if option == CREATE_NEW_TOPIC_OPTION:
        return "➕ Crear un nuevo tema de análisis..."
    return f"📁 {option.replace('_', ' ').title()}"

# Widget para seleccionar el tema
selected_topic = st.selectbox(
    "Selecciona un tema para trabajar o crea uno nuevo:",
    options=topic_options,
    index=active_topic_index,
    format_func=format_topic_option,
    key="topic_selector"
)

# Lógica para manejar la selección del usuario
if selected_topic == CREATE_NEW_TOPIC_OPTION:
    # Si el usuario quiere crear un nuevo tema, mostrar el asistente
    st.info("Se iniciará el asistente para configurar un nuevo tema.")
    if st.button("🚀 Iniciar Asistente de Configuración", type="primary"):
        st.session_state.mostrar_wizard = True
        st.rerun()
elif selected_topic and selected_topic != ACTIVE_TOPIC:
    # [FIX] Limpiar el estado de la sesión para forzar la recarga de datos del nuevo tema
    keys_to_clear = [
        'df_review', 'topic_name_for_review',
        'df_active_review', 'topic_name_for_active_review',
        'df_finetune', 'topic_name_for_finetune',
        'keywords', 'topic_name_for_keywords',
        'exclusions', 'topic_name_for_exclusions',
        'available_columns'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Actualizar la configuración con el nuevo tema
    config["ACTIVE_TOPIC"] = selected_topic
    from utils import save_config
    save_config(config)
    st.toast(f"✅ Tema cambiado a: {selected_topic.upper()}", icon="🔄")
    st.rerun()
else:
    # Si el tema activo es el seleccionado, mostrar el estado actual
    if ACTIVE_TOPIC:
        st.success(f"Tema activo: **{ACTIVE_TOPIC.upper()}**")
    else:
        st.warning("No hay ningún tema activo. Por favor, selecciona uno o crea uno nuevo.")

st.markdown("---")
# Crear botones de navegación principales
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### ⚙️ Configuración")
    st.markdown("Gestionar temas, archivos y parámetros")
    if st.button("⚙️ Configuración de Busqueda", type="primary", use_container_width=True):
        st.switch_page("pages/1_Configuracion_de_Busqueda.py")
with col2:
    st.markdown("### 🎛️ Panel de Control")
    st.markdown("Ejecutar procesos y flujos de trabajo")
    if st.button("🎛️ Panel de Control", type="primary", use_container_width=True):
        st.switch_page("pages/2_Panel_de_Control.py")
with col3:
    st.markdown("### 🏠 Estado del Proyecto")
    st.markdown("Estado del proyecto actual")
    if st.button("🏠 Ir a Estado del Proyecto", type="primary", use_container_width=True):
        st.switch_page("pages/3_Estado_del_Proyecto.py")
st.markdown("---")
# Mostrar wizard si es necesario
mostrar_boton_wizard()

# Agregar ayuda contextual con tooltips
st.info("💡 Esta es la página principal para usar el sistema es necesario configurar los parámetros de análisis y seguir el flujo de trabajo para clasificar contratos. Puedes navegar por las diferentes secciones usando el menú de la izquierda.")

st.markdown("---")

st.markdown("#### 💡 Beneficios clave:")
st.markdown("""
- **Eficiencia:** Reduce drásticamente el tiempo necesario para encontrar contratos relevantes.
- **Precisión:** Identifica contratos que podrían pasar desapercibidos con métodos tradicionales.
- **Consistencia:** Aplica un criterio uniforme en la clasificación de todos los documentos.
- **Adaptabilidad:** Mejora continuamente gracias a la retroalimentación del usuario.
""")

# --- Ayuda y Documentación ---
st.markdown("## 📚 Documentación y Ayuda")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📖 Guía de Uso")
    st.markdown("Aprende cómo funciona el sistema paso a paso")
    if st.button("📚 Ver Metodología", use_container_width=True):
        st.switch_page("pages/5_Guia_de_Uso.py")

with col2:
    st.markdown("### 📊 Historial de Modelos")
    st.markdown("Revisa el rendimiento de modelos anteriores")
    if st.button("📈 Ver Historial", use_container_width=True):
        st.switch_page("pages/4_Historial_de_Modelos.py")
        
# --- Pie de página ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    <p>Sistema de Clasificación Inteligente de Contratos v1.0.0</p>
    <p>Desarrollado por MCS Año 2025 - GAEC/EMAE - Colombia Compra Eficiente</p>
</div>
""", unsafe_allow_html=True)