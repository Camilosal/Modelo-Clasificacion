import streamlit as st
from utils import (
    get_active_topic, crear_dashboard_estado, mostrar_dashboard_visual, 
    mostrar_boton_wizard, load_config, mostrar_wizard_configuracion
)

st.set_page_config(
    page_title="Panel de Estado del Proyecto",
    page_icon="📊",
    layout="wide"
)

# --- Logo de la Entidad ---
st.sidebar.image("assets/logo.png")

# --- Sección "Acerca de" en la barra lateral ---
st.sidebar.markdown("---")
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

# --- Interfaz Principal ---
# --- Cargar configuración ---
config = load_config()
ACTIVE_TOPIC = config.get("ACTIVE_TOPIC", "")

# --- Estado del Proyecto ---
st.markdown("#### 🚀 Espera a que carge toda la pagina, el sistema esta comprobando el estado de ejecución")
if ACTIVE_TOPIC:
    st.markdown(f"## 📊 Estado del Proyecto: **{ACTIVE_TOPIC.upper()}**")
    dashboard = crear_dashboard_estado(ACTIVE_TOPIC)
    mostrar_dashboard_visual(dashboard)
    
    # Mostrar siguiente acción recomendada
    st.markdown("### 🎯 Próxima Acción Recomendada")
    st.info(f"**{dashboard['siguiente_accion']}**")
    
    # Botones de acción rápida basados en el estado
    st.markdown("### 🚀 Acciones Rápidas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if dashboard['pasos_completados'] == 0:
            if st.button("🔧 Configurar Proyecto", type="primary", use_container_width=True):
                st.switch_page("pages/1_Configuracion_de_Busqueda.py")
        else:
            if st.button("🎛️ Panel de Control", type="primary", use_container_width=True):
                st.switch_page("pages/2_Panel_de_Control.py")
    
    with col2:
        if dashboard['pasos_completados'] > 0:
            # Determinar el siguiente paso
            if dashboard['pasos_completados'] == 1:
                if st.button("2️⃣ Validación Humana", use_container_width=True):
                    st.switch_page("pages/7_PASO_2_Validacion_Humana.py")
            elif dashboard['pasos_completados'] == 2:
                if st.button("3️⃣ Aprender y Refinar", use_container_width=True):
                    st.switch_page("pages/8_PASO_3_Aprender_y_Refinar.py")
            elif dashboard['pasos_completados'] == 3:
                if st.button("4️⃣ Entrenar Clasificador", use_container_width=True):
                    st.switch_page("pages/9_PASO_4_Entrenar_Clasificador.py")
            elif dashboard['pasos_completados'] == 4:
                if st.button("5️⃣ Clasificar Masivamente", use_container_width=True):
                    st.switch_page("pages/10_PASO_5_Clasificar_con_Predicciones.py")
            else:
                if st.button("📈 Ver Resultados", use_container_width=True):
                    st.switch_page("pages/4_Historial_de_Modelos.py")
        else:
            if st.button("🧙‍♂️ Asistente de Configuración", use_container_width=True):
                st.session_state.mostrar_wizard = True
                st.rerun()
    
    with col3:
        if st.button("⚙️ Configuración", use_container_width=True):
            st.switch_page("pages/1_Configuracion_de_Busqueda.py")
            
else:
    st.markdown("## 📊 Estado del Proyecto")
    st.info("📋 **No hay tema activo.** Crea tu primer proyecto para ver el dashboard.")
    
    st.markdown("### 🚀 Comenzar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🧙‍♂️ Configuración Guiada**")
        st.markdown("Ideal para usuarios nuevos")
        if st.button("🚀 Iniciar Asistente", type="primary", use_container_width=True):
            st.session_state.mostrar_wizard = True
            st.rerun()
    
    with col2:
        st.markdown("**⚙️ Configuración Manual**")
        st.markdown("Para usuarios experimentados")
        if st.button("🔧 Configuración Avanzada", use_container_width=True):
            st.switch_page("pages/1_Configuracion_de_Busqueda.py")

# Mostrar wizard si está activo
if st.session_state.get('mostrar_wizard', False):
    st.markdown("---")
    mostrar_wizard_configuracion()
    
    if st.button("❌ Cerrar Asistente", type="secondary"):
        st.session_state.mostrar_wizard = False
        st.rerun()
