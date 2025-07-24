import streamlit as st

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Guía de Uso del Sistema",
    page_icon="📖",
    layout="wide"
)

st.title("📖 Guía de Uso del Sistema de Clasificación de Contratos")

st.markdown("""
¡Bienvenidx!

Este sistema está diseñado como un **proceso de mejora continua** donde tu experiencia en el tema a analizar ayuda a que el programa aprenda y se vuelva más preciso. El objetivo es clasificar contratos de manera exacta, adaptándose al tema específico que necesites.

# Imagen del banner del sistema con opción de ampliar
st.markdown("""
### 🎯 Descripción General del Sistema
""")

st.image(
    "assets/banner_sistema.png",
    caption="Click en la imagen para ampliarla.",
    width=400
)
El proceso se maneja principalmente desde el **🎛️ Panel de Control** y se divide en **tres grandes etapas**.
""")

st.markdown("---")

# --- Parte 1: Arquitectura y Configuración ---
st.header("Parte 1: Cómo Funciona el Sistema y Configuración Inicial")

with st.expander("A. Entendiendo Cómo Obtiene los Datos", expanded=True):
    st.markdown("""
    El sistema cuenta con una manera flexible de obtener datos. La carga sigue un orden automático para asegurar que siempre tengas la información más actualizada:

    **Orden de Prioridad: API → Base de Datos → Archivo Local (Excel/CSV)**

    1. **Primer Intento: Conexión API.** El sistema primero intentará conectarse a la **API** que hayas configurado. Si funciona, usará estos datos.
    
    > **Nota para APIs tipo Socrata:** Puedes incluir una consulta **SoQL** en la configuración de la API para pre-filtrar o seleccionar columnas específicas directamente desde la fuente de datos.

    2. **Segundo Intento: Base de Datos.** Si la API falla o no está configurada, el sistema intentará conectarse a la **base de datos**.

    3. **Último Recurso: Archivo Local.** Si tanto la API como la base de datos fallan, el sistema usará el **archivo Excel o CSV** que hayas subido como respaldo.

    Sin importar de dónde vengan, los datos se procesan y se guardan en un formato estándar que será usado por todos los componentes del sistema para garantizar consistencia.
    """)

with st.expander("B. Configuración por Primera Vez: Ayuda Paso a Paso"):
    st.markdown("""
    La forma más fácil de empezar es usando la **Ayuda de Configuración Paso a Paso**, disponible en la página de **⚙️ Configuración de Búsqueda**.

    La ayuda te guiará a través de estos pasos:

    **Paso 1: Creación del Tema.** Defines el nombre de tu proyecto (ej. `ciberseguridad`).

    **Paso 2: Configuración de Fuentes de Datos.** Introduces los detalles de conexión para la API y/o la base de datos (ambos opcionales) o subes tu archivo CSV con los datos.

    **Paso 3: Personalización de la Búsqueda.** Añades tus palabras clave iniciales, términos de exclusión, codigos UNSPSC a filtrar (Para enfocar los esfuerzos en los contratos o procesos relevantes).

    **Paso 4: Resumen y Finalización.** Revisas toda la configuración y guardas tu nuevo proyecto.
    """)

with st.expander("C. Configuración Manual y Avanzada"):
    st.markdown("""
    Si prefieres no usar la ayuda automática, puedes configurar todo manualmente desde la página de **⚙️ Configuración de Búsqueda**:

    - **Gestionar Temas:** Crea, selecciona o elimina los temas de tu proyecto.
    - **Personalizar Proyecto:** Configura los parámetros de búsqueda para el tema activo (Palabras Clave, Exclusiones, Códigos UNSPSC, Columnas de Texto).
        - **Sé específico con las palabras clave:** Combina términos técnicos y coloquiales
        - **Investiga a profundidad:** Usa fuentes confiables para definir tus palabras clave, incluye diversidad de fuentes y pregunta o asesorate de conocedores del tema
    - **Fuente de Datos:** Configura los detalles de conexión para API, base de datos o asigna un archivo local.
    """)

st.markdown("---")

# --- Parte 2: El Proceso de Clasificación ---
st.header("Parte 2: El Proceso de Trabajo desde el Panel Principal")

st.info("El **🎛️ Panel Principal** es tu centro de operaciones. Desde aquí ejecutarás todo el proceso de trabajo.")

# Imagen del proceso completo
st.markdown("""
### 🗺️ Pasos Completos del Proceso
""")

st.image(
    "assets/pasos_completos.png",
    caption="Click en la imagen para ampliarla.",
    width=350
)

with st.expander("Etapa 1: Generación de Candidatos y Validación Inicial", expanded=True):
    st.markdown("""
    ### 🚀 Paso 1: Generar Candidatos para Revisión
    En el **🎛️ Panel de Control**, antes de hacer clic en **`🚀 Generar Candidatos`**, asegúrate de configurar las columnas correctas en los selectores que aparecen justo debajo del botón. Debes indicar qué columnas contienen el **texto a analizar** y cuál contiene los **códigos UNSPSC**.

    **¿Qué hace el sistema internamente?**
    - Lee el archivo de datos estandarizado (`datos_preprocesados.parquet`).
    - Filtra los datos usando la columna y los códigos UNSPSC que definiste, manejando inteligentemente diferentes formatos de código.
    - Realiza una limpieza de texto masiva, aprovechando **todos los procesadores de tu equipo** para acelerar el proceso.
    - Busca contratos que contengan tus palabras clave (ya limpias y normalizadas).
    - Utiliza búsqueda semántica inteligente para encontrar contratos relacionados. Para esto, **guarda en caché los análisis de texto (embeddings)**, por lo que las ejecuciones futuras serán mucho más rápidas.
    - Te presenta los candidatos para revisar en un archivo Excel ligero, que contiene solo las columnas más importantes.

    **Tiempo estimado:** 60-180 minutos según el tamaño de tus datos.

    ### ✅ Paso 2: Validar Datos Manualmente
    Una vez generados los candidatos, ve a la página **`📝 PASO 2: Validación Humana`**. Esta es tu tarea más importante.

    **¿Cómo funciona la validación?**
    - El sistema es inteligente y **solo te mostrará los contratos que no has validado antes**.
    - Tu tarea es leer la descripción y marca cada contrato preseleccionado como `SÍ` (relevante) o `NO` (no relevante).
    - Al guardar, el sistema archiva tu trabajo y actualiza el historial para no volver a mostrarte los mismos ítems.

    **Opciones de Validación:**
    - **Validación en la App:** Usa la tabla interactiva directamente en la página.
    - **Validación Externa:** En el **🎛️ Panel de Control**, en la sección del Paso 2, puedes **descargar el archivo de revisión**, editarlo en Excel y **volver a subirlo**. El sistema se actualizará automáticamente.
    """)

with st.expander("Etapa 2: Aprendizaje y Clasificación Masiva", expanded=True):
    st.markdown("""
    ### 🤖 Paso 3: Aprender de ti y Entrenar Modelos
    De vuelta en el Panel Principal, haz clic en **`🤖 Ejecutar Ciclo de Aprendizaje`**. Este botón ahora desencadena un sofisticado proceso de aprendizaje en varios pasos:

    **¿Qué sucede durante el aprendizaje?**
    1.  **Análisis de Keywords:** El sistema analiza el rendimiento de tus palabras clave actuales y genera un reporte con sugerencias de nuevos términos basados en los contratos que marcaste como `SI`.
    2.  **Creación del Dataset de Estudio:** Usando tus validaciones, el sistema crea automáticamente un "material de estudio" para el modelo experto. Genera pares de textos `SI` vs `SI` (para enseñar similitud) y `SI` vs `NO` (para enseñar diferencias).
    3.  **Entrenamiento del Modelo Experto (Fine-Tuning):** El sistema toma el modelo base especializado en español (`hiiamsid/sentence_similarity_spanish_es`) y lo entrena con el dataset de estudio. Esto lo convierte en un verdadero experto en la jerga y contexto de tu tema.
    4.  **Entrenamiento del Clasificador Final:** Finalmente, se entrena el modelo que toma la decisión final. Este modelo utiliza no solo el análisis de texto del modelo experto, sino también cualquier **columna categórica** (ej. 'Modalidad de Contratación') que hayas seleccionado en el Panel de Control para una precisión aún mayor.

    **Tiempo estimado:** 60-240 minutos según la cantidad de validaciones

    ### 📊 Paso 4: Aplicar Modelo y Generar Reporte
    Con los modelos ya entrenados, haz clic en **`📊 Ejecutar Clasificación`** en el Panel de Control.

    **El momento de la verdad:**
    - Antes de ejecutar, puedes **ajustar el umbral de confianza** con el slider. Solo los contratos que superen este umbral serán marcados como `SI`.
    - El sistema aplica todo el conocimiento mejorado con lo que le enseñaste a la base de datos completa.
    - Clasifica automáticamente todos los contratos.
    - Asigna un nivel de confianza a cada predicción.
    - Genera reportes detallados que puedes descargar.
    """)

with st.expander("Etapa 3: Calibración y Revisión Activa", expanded=True):
    st.markdown("""
    ### 🎯 La Etapa de Perfeccionamiento del Sistema

    Esta es la fase más potente del sistema, donde refinas el modelo para alcanzar la máxima precisión. Aquí el sistema se vuelve realmente inteligente al identificar exactamente qué casos necesita que valides para mejorar más rápido.

    ---

    ### 🎯 Paso 5: Generar Revisión Activa (Active Learning)
    En el Panel Principal, haz clic en **`🎯 Generar Revisión Activa`**. El sistema ya no selecciona contratos al azar, sino que busca específicamente los casos más **inciertos** para el modelo.

    **¿Cómo funciona la Revisión Activa?**
    - **Contratos en la "Zona Gris":** Se seleccionan contratos donde la confianza del modelo para decir 'SI' está en un rango configurable (por defecto, entre 70% y 90%). Estos son los casos donde el modelo más duda y donde tu feedback es más valioso para refinar sus límites de decisión.
    - **Tamaño optimizado:** El sistema te presenta un número manejable de contratos (configurable, por defecto 250) para que tu revisión sea enfocada y de alto impacto.

    **Tiempo estimado:** 2-5 minutos para generar la lista

    ---

    ### ✅ Paso 6: Validar la Revisión Activa (2º Ciclo)
    Ve de nuevo a la página **`📝 PASO 2: Validación Humana`** y selecciona la pestaña **`Revisión Activa (2º Ciclo)`**.

    **¿Por qué esta validación es más valiosa?**
    - Cada contrato que valides aquí tiene **5-10 veces más impacto** que una validación normal
    - Estás enseñando al sistema en sus "puntos débiles"
    - Ayudas a definir mejor los límites entre contratos "SÍ" y "NO"
    - Reduces significativamente los errores en la próxima clasificación

    **Mejores prácticas para Revisión Activa:**
    - **Sé especialmente cuidadoso:** Estos casos son más complejos por naturaleza
    - **Usa comentarios generosamente:** Explica tu razonamiento en casos difíciles
    - **Mantén consistencia:** Aplica los mismos criterios que en la primera ronda
    - **Valida al menos 80%** de la lista para habilitar la mejora automática

    **Indicadores de progreso:**
    El sistema te mostrará tu progreso: "Validadas: 145/600 (75%)" - necesitas llegar al 80% para continuar.
    O descarga el archivo de revisión, valida en Excel y vuelve a subirlo.

    ---

    ### 🚀 Paso 7: Iniciar Ciclo de Mejora
    **¡El paso final y más poderoso!** Regresa al Panel Principal.

    **Cuando esté disponible:**
    Verás que el botón **`🚀 Iniciar Ciclo de Mejora`** está habilitado (solo si has validado más del 80% de la revisión activa).

    **¿Qué hace el Ciclo de Mejora Automática?**
    Al hacer clic, el sistema ejecuta **automáticamente** todo el proceso de nuevo, pero con toda tu información adicional:

    1. **Consolida** todas tus validaciones (primera ronda + revisión activa)
    2. **Re-analiza** el rendimiento de las palabras clave con los nuevos ejemplos
    3. **Re-entrena** los modelos con el conocimiento expandido
    4. **Re-calibra** los niveles de confianza
    5. **Re-clasifica** toda la base de datos con el modelo mejorado
    6. **Genera** nuevos reportes comparando el rendimiento anterior

    **Tiempo estimado:** 60-320 minutos completamente automático

    **Resultados esperados:**
    - **Mejora en precisión:** +10-20% en la primera mejora
    - **Mejor confianza:** Predicciones más seguras y confiables
    - **Menos falsos positivos:** Resultados más exactos
    - **Mejor cobertura:** Encuentra contratos que antes se escapaban

    ---

    ### 🔄 Ciclos Adicionales de Mejora
    **¡El proceso es repetible!** Una vez completado el primer ciclo de mejora:

    - Puedes ejecutar **nuevos ciclos** de revisión activa
    - Cada ciclo adicional produce **mejoras incrementales**
    - Típicamente, 2-3 ciclos son suficientes para alcanzar **>95% de precisión**
    - El sistema **aprende continuamente** de cada validación que hagas

    **¿Cuándo parar?**
    - Cuando la precisión se estabilice (cambios <2% entre ciclos)
    - Cuando estés satisfecho con los resultados
    - Cuando las nuevas revisiones activas requieran pocas correcciones

    ### 📊 Seguimiento del Progreso
    El sistema te mostrará **métricas comparativas** entre ciclos:

    | Métrica | Inicial | Después 1er Ciclo | Después 2do Ciclo |
    |---------|---------|-------------------|-------------------|
    | Precisión | 75% | 89% | 94% |
    | Cobertura | 82% | 91% | 93% |
    | F1-Score | 0.78 | 0.90 | 0.94 |
    | Contratos Identificados | 450 | 523 | 547 |

    **¡Has completado un ciclo de mejora!** Los resultados de esta nueva clasificación serán significativamente más precisos.
    """)

st.markdown("---")

# --- Parte 3: Gestión Avanzada - Copias de Seguridad y Restauración ---
st.header("Parte 3: Gestión Avanzada - Copias de Seguridad y Restauración")

with st.expander("A. Cómo Crear una Copia de Seguridad"):
    st.markdown("""
    Esta funcionalidad te permite crear un respaldo completo de un tema específico, incluyendo su configuración, palabras clave, modelos entrenados y, lo más importante, tu trabajo de validación manual.

    **¿Cuándo deberías hacer una copia de seguridad?**
    - Antes de hacer cambios importantes en las palabras clave.
    - Antes de restaurar una copia de seguridad antigua.
    - Periódicamente, para tener un punto de restauración seguro.

    **¿Cómo se hace?**
    1.  Ve a la página de **⚙️ Configuración de Búsqueda**.
    2.  En la sección "Gestionar Temas de Análisis", asegúrate de que el tema que quieres respaldar esté **activo**.
    3.  Busca la sub-sección **💾 Copia de Seguridad del Tema Activo**.
    4.  Haz clic en el botón **`📦 Crear Copia de Seguridad para '<tu_tema>'`**.

    El sistema creará una carpeta con fecha y hora dentro del directorio `backups/` en la raíz de tu proyecto.
    """)

with st.expander("B. Cómo Restaurar una Copia de Seguridad (Proceso Manual)"):
    st.warning("""
    **¡MUY IMPORTANTE! PRECAUCIÓN**

    Restaurar una copia de seguridad **sobrescribirá la configuración y los modelos actuales** del tema en tu proyecto. Si quieres guardar el estado actual antes de restaurar, haz una copia de seguridad de ese tema primero.
    """
    )
    st.markdown("""
    Restaurar una copia de seguridad es un proceso **manual** que consiste en copiar los archivos del respaldo a su lugar original.

    #### **Paso 1: Localiza tu Copia de Seguridad**
    1.  En la carpeta principal de tu proyecto, busca y abre la carpeta `backups/`.
    2.  Encuentra la carpeta del respaldo que quieres restaurar (ej: `backup_ciberseguridad_20250720_103000`).
    3.  Abre esa carpeta para ver todos los archivos respaldados.

    #### **Paso 2: Copia y Reemplaza los Archivos**
    Copia cada archivo o carpeta desde tu backup a su ubicación original en el proyecto. El sistema operativo te preguntará si quieres reemplazar los archivos existentes; debes **confirmar que sí**.

    Usa esta tabla como guía:

    | Archivo/Carpeta en la Copia de Seguridad | 📂 **Destino Final en tu Proyecto** |
    | :--- | :--- |
    | `config.json` | Carpeta principal del proyecto (`/`) |
    | `keywords_<tema>.xlsx` | `archivos_entrada/` |
    | `exclusion_words_<tema>.xlsx` | `archivos_entrada/` |
    | `contratos_para_revision_humana...` | `resultados/` |
    | `clasificador_<tema>_v1.joblib` | `resultados/` |
    | `reporte_clasificacion_<tema>.json`| `resultados/` |
    | `historial_entrenamiento_<tema>.csv`| `resultados/` |
    | Carpeta `<tema>-experto-v1` | `resultados/modelos_afinados/` |

    #### **Paso 3: Reinicia y Verifica**
    1.  **Detén y vuelve a iniciar la aplicación de Streamlit.** Este paso es crucial para que el sistema cargue el archivo `config.json` que acabas de restaurar.
    2.  Una vez reiniciada, ve a la página de "Configuración de Búsqueda". El tema que restauraste debería estar activo y su configuración (palabras clave, descripción, etc.) debería reflejar el estado del backup.
    """)

st.markdown("---")

# --- Glosario Completo ---
st.header("📚 Glosario Completo de Términos")

with st.expander("🎯 Conceptos Fundamentales del Sistema"):
    st.markdown("""
    ### Términos Básicos

    **Tema**
    - El área específica de interés para la clasificación (ej. 'ciberseguridad', 'infraestructura')
    - Cada tema es un proyecto independiente con su propia configuración
    - Puedes tener múltiples temas y trabajar con ellos por separado

    **Palabras Clave (Keywords)**
    - Términos que definen el punto de partida de la búsqueda
    - Son la base para encontrar los primeros candidatos
    - Ejemplo: ["firewall", "antivirus", "seguridad informática"]
    - Se mejoran automáticamente basado en tus validaciones

    **Términos de Exclusión**
    - Lista de palabras que ayudan a filtrar contratos irrelevantes
    - Útiles para eliminar falsos positivos obvios
    - Ejemplo: ["limpieza", "cafetería", "papelería"] para búsquedas tecnológicas, usalo con cuidado para no eliminar contratos relevantes o no entrenar al modelo para identificar sezgos

    **Candidatos**
    - Lista inicial de contratos que el sistema considera potencialmente relevantes
    - Generados usando palabras clave y búsqueda inteligente
    - Requieren validación humana para confirmar si son correctos o no
    """)

with st.expander("🤖 Inteligencia Artificial y Tecnología"):
    st.markdown("""
    ### Tecnologías del Sistema

    **Búsqueda Semántica**
    - Método de búsqueda inteligente que entiende el significado, no solo palabras exactas
    - Encuentra contratos relacionados por concepto, no solo por términos idénticos
    - Ejemplo: puede encontrar "protección informática" cuando buscas "ciberseguridad"

    **Modelo de Inteligencia Artificial**
    - Programa que aprende patrones de tus validaciones
    - Se entrena específicamente con tu conocimiento experto
    - Mejora automáticamente cada vez que validas más contratos

    **Clasificador Automático**
    - El componente que toma decisiones "SÍ" o "NO" para cada contrato
    - Entrenado exclusivamente con tus validaciones
    - Capaz de procesar miles de contratos en minutos

    **Nivel de Confianza**
    - Porcentaje (0% a 100%) que indica qué tan seguro está el sistema de su predicción
    - **90-100%:** Muy seguro - probablemente correcto
    - **70-89%:** Moderadamente seguro - revisar si es crítico
    - **50-69%:** Inseguro - recomendable validación manual
    - **0-49%:** Muy inseguro - requiere validación humana obligatoria
    """)

with st.expander("📊 Métricas y Evaluación de Rendimiento"):
    st.markdown("""
    ### Entendiendo las Métricas

    **Precisión**
    - De todos los contratos que el sistema marcó como "SÍ", ¿qué porcentaje realmente lo era?
    - **Alta precisión = pocos errores de falsos positivos**
    - Ejemplo: Si el sistema dice "SÍ" a 100 contratos y 90 realmente lo son, la precisión es 90%

    **Cobertura (Recall)**
    - De todos los contratos que realmente son "SÍ", ¿qué porcentaje encontró el sistema?
    - **Alta cobertura = no se pierden contratos importantes**
    - Ejemplo: Si hay 200 contratos relevantes y el sistema encuentra 180, la cobertura es 90%

    **F1-Score**
    - Una métrica que combina Precisión y Cobertura en un solo número
    - Rango de 0 a 1, donde 1 es perfecto
    - **0.90-1.00:** Excelente - listo para uso en producción
    - **0.80-0.89:** Bueno - usar con supervisión ocasional
    - **0.70-0.79:** Aceptable - necesita más ciclos de mejora
    - **Menos de 0.70:** Requiere revisión de configuración

    **Falsos Positivos**
    - Contratos que el sistema marca como "SÍ" pero realmente son "NO"
    - Problema: Te hace revisar contratos irrelevantes
    - Se reduce mejorando la precisión

    **Falsos Negativos**
    - Contratos que el sistema marca como "NO" pero realmente son "SÍ"
    - Problema: Se pierden contratos importantes
    - Se reduce mejorando la cobertura
    """)

with st.expander("🔄 Procesos y Metodología"):
    st.markdown("""
    ### Procesos Clave

    **Validación Humana**
    - El proceso donde tú, como experto, revisas contratos y decides si son relevantes
    - Es la parte más importante - tu conocimiento entrena al sistema
    - Debe ser consistente y cuidadosa para obtener buenos resultados

    **Aprendizaje Automático Supervisado**
    - Técnica donde el sistema aprende de ejemplos que tú proporcionas
    - "Supervisado" significa que tú le enseñas las respuestas correctas
    - El sistema encuentra patrones en tus decisiones y los aplica a nuevos casos

    **Revisión Activa**
    - Estrategia inteligente que prioriza los contratos más informativos para validar
    - En lugar de casos aleatorios, te muestra los que más ayudarán a mejorar el sistema
    - Maximiza el impacto de tu tiempo de validación

    **Ciclo de Mejora Continua**
    - Proceso iterativo: validar → entrenar → aplicar → revisar → mejorar
    - Cada ciclo incrementa la precisión del sistema
    - Diseñado para converger rápidamente a alta calidad

    **Entrenamiento/Re-entrenamiento**
    - Proceso donde el sistema actualiza su conocimiento con nuevas validaciones
    - Ocurre cada vez que ejecutas un ciclo de aprendizaje
    - Permite que el sistema se adapte a nuevos patrones y mejore continuamente
    """)

with st.expander("🛠️ Aspectos Técnicos Simplificados"):
    st.markdown("""
    ### Configuración y Datos

    **API (Interfaz de Programación)**
    - Conexión directa con sistemas externos para obtener datos automáticamente
    - Permite tener información actualizada sin subir archivos manualmente
    - Requiere configuración técnica inicial pero automatiza el proceso

    **Base de Datos SQL**
    - Sistema de almacenamiento estructurado para grandes cantidades de datos
    - Permite consultas específicas y filtrado avanzado
    - Alternativa a archivos para organizaciones con sistemas de información robustos

    **(CSV/Excel)**
    - Formato de respaldo más simple y confiable
    - Siempre requerido aunque tengas API o base de datos configuradas
    - Fácil de preparar y subir desde cualquier computador

    **Preprocesamiento**
    - Limpieza y preparación automática de datos antes del análisis
    - Incluye: normalización de texto, eliminación de caracteres especiales, etc.
    - Mejora la calidad y consistencia para el procesamiento posterior

    **Procesamiento por Lotes**
    - Técnica para manejar grandes cantidades de datos eficientemente
    - El sistema procesa los contratos en grupos pequeños para optimizar memoria
    - Permite trabajar con bases de datos de cientos de miles de contratos

    ### Archivos y Resultados

    **Datos Preprocesados**
    - Versión limpia y estandarizada de tus datos originales
    - Se genera automáticamente y se usa internamente por el sistema
    - Garantiza consistencia en todo el proceso de análisis

    **Reportes de Clasificación**
    - Archivos Excel/CSV con todos los resultados del sistema
    - Incluyen: decisión (SÍ/NO), nivel de confianza, y datos originales del contrato
    - Descargables al final del proceso para usar en otros sistemas

    **Métricas de Rendimiento**
    - Estadísticas detalladas sobre qué tan bien está funcionando el sistema
    - Incluyen gráficos y tablas comparativas entre diferentes ciclos
    - Te ayudan a decidir si necesitas más validación o si el sistema está listo
    """)

st.markdown("---")

# --- Consejos Finales ---
st.header("💡 Consejos para el Éxito")

with st.expander("🎯 Mejores Prácticas para Obtener Excelentes Resultados"):
    st.markdown("""
    ### Durante la Configuración Initial
    - **Sé específico con las palabras clave:** Combina términos técnicos y coloquiales
    - **Investiga a profundidad:** Usa fuentes confiables para definir tus palabras clave, incluye diversidad de fuentes y pregunta o asesorate de conocedores del tema
    - **Usa exclusiones inteligentemente:** Es mejor filtrar demasiado al inicio, usalo con cuidado para no eliminar contratos relevantes o no entrenar al modelo para identificar sezgos
    - **Prepara datos de calidad:** Limpia tu archivo Excel antes de subirlo

    ### Durante la Validación
    - **Mantén criterios consistentes:** Define reglas claras y síguelas
    - **Documenta casos dudosos:** Usa comentarios para explicar decisiones difíciles
    - **Valida suficientes ejemplos:** Mínimo 1000, idealmente 200-300 nuevos por ciclo
    - **Distribuye bien los ejemplos:** Incluye variedad de entidades y tipos de contrato

    ### Durante los Ciclos de Mejora
    - **Sé paciente:** La mejora es gradual pero consistente
    - **Revisa las métricas:** Observa las tendencias, no solo los números absolutos
    - **No sobre-entrenes:** 2-3 ciclos suelen ser suficientes para la mayoría de casos

    ### Señales de que el Sistema está Listo
    - F1-Score consistente por encima de 0.85 por 2+ ciclos
    - Precisión y Cobertura balanceadas (diferencia menor a 10%)
    - Nivel de confianza promedio superior al 75%
    - Las nuevas validaciones requieren menos del 10% de correcciones
    """)

st.markdown("---")

st.success("""
🎉 **¡Felicitaciones!** Has completado la guía completa del Sistema de Clasificación de Contratos.

**Próximos Pasos Recomendados:**
1. 🎯 Ve a **⚙️ Configuración** y usa la Ayuda Paso a Paso
2. 🚀 Ejecuta tu primer ciclo desde el **🎛️ Panel Principal**  
3. 📝 Valida cuidadosamente tus primeros candidatos
4. 🔄 Usa la Revisión Activa para perfeccionar el sistema
5. 🚀 ¡Ejecuta ciclos de mejora hasta alcanzar la precisión deseada!

**Soporte:** ¿Tienes preguntas? Revisa el glosario de términos o contacta al equipo técnico.
""")