# Sistema de Clasificación Inteligente de Contratos

Este sistema es una herramienta de inteligencia artificial diseñada para analizar, clasificar y aprender de grandes volúmenes de contratos. Utiliza un ciclo de **mejora continua** donde el conocimiento de un experto humano se utiliza para entrenar y refinar modelos de IA, logrando una clasificación cada vez más precisa y adaptada a la jerga de un tema específico.

El objetivo principal es automatizar la tediosa tarea de revisión manual de contratos, permitiendo al usuario encontrar documentos relevantes de manera eficiente y consistente.

## Metodología Principal

El sistema se basa en un ciclo virtuoso de 4 pasos que se retroalimentan:

1.  **🚀 GENERAR:** El sistema busca en una base de datos de contratos y genera una lista de "candidatos" que podrían ser relevantes para un tema, usando una combinación de **búsqueda por palabras clave** y **búsqueda semántica** con IA.
2.  **📝 VALIDAR:** Un **experto humano** revisa la lista de candidatos y los marca como `SI` (relevante) o `NO` (no relevante). Este es el paso más crucial, ya que la calidad de la validación determina la calidad del aprendizaje.
3.  **🧠 APRENDER:** El sistema utiliza las validaciones del experto para ser más inteligente. Esto puede implicar:
    *   Analizar qué palabras clave son más efectivas.
    *   Sugerir nuevas palabras clave.
    *   Entrenar un **modelo de lenguaje experto** (fine-tuning) que entienda el contexto específico del tema.
4.  **🏆 REPETIR Y CLASIFICAR:** Con la inteligencia mejorada, el ciclo se repite para encontrar mejores candidatos. Finalmente, se entrena un **modelo clasificador** que puede aplicarse a toda la base de datos para obtener una clasificación final y completa.

---

## Características Principales

- **Interfaz Web Intuitiva:** Una aplicación Streamlit que guía al usuario paso a paso a través de todo el flujo de trabajo.
- **Gestión Multi-Tema:** Permite trabajar en diferentes proyectos o "temas" (ej. 'ciberseguridad', 'obra civil') de forma independiente.
- **Estructura Organizada:** Separa claramente los datos de entrada (`archivos_entrada`), los resultados (`resultados`) y los scripts de lógica.
- **Doble Motor de Búsqueda:** Combina la precisión de las **palabras clave** con la flexibilidad de la **búsqueda semántica**.
- **Ciclos de Mejora Automatizados:**
    - **Validación Inteligente:** El sistema presenta únicamente los contratos que no han sido revisados previamente, ahorrando tiempo al experto.
    - **Consolidación de Entrenamiento:** Antes de aprender, el sistema consolida todas las validaciones históricas, usando siempre la versión más reciente de la opinión del experto.
    - **Re-entrenamiento Incremental:** El modelo experto no empieza de cero. Carga su conocimiento previo y lo refina con los nuevos datos, mejorando continuamente.
- **Modelo Experto (Fine-Tuning):** Capacidad avanzada para especializar un modelo de lenguaje, con optimizaciones de **Precisión Mixta (AMP)** para un entrenamiento hasta un 60% más rápido en GPUs NVIDIA.
- **Ciclo de Revisión Activa:** Identifica los casos más dudosos para el modelo y los presenta al experto para una revisión prioritaria.
- **Estado del Proyecto:** Ofrece visualizaciones claras sobre el rendimiento de los modelos y la efectividad de las palabras clave.
- **Auditoría y Trazabilidad:** Todas las validaciones se guardan con fecha y hora en un historial, permitiendo una auditoría completa del proceso de decisión.

---

## ⚙️ Instalación y Configuración

Asegúrate de tener **Python 3.9 o superior** instalado en tu sistema.

1.  **Clona el repositorio:**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_CARPETA_PROYECTO>
    ```

2.  **Crea y activa un entorno virtual (Recomendado):**
    ```bash
    python -m venv venv
    # En Windows
    venv\Scripts\activate
    # En macOS/Linux
    source venv/bin/activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r paquetes.txt
    ```

4.  **Descarga el modelo de lenguaje en español:**
    El sistema utiliza `spaCy` para el procesamiento de texto. Descarga el modelo necesario con el siguiente comando:
    ```bash
    python -m spacy download es_core_news_lg
    ```

---

## 🚀 Guía de Uso Rápido (Interfaz Web)

La forma más sencilla de utilizar el sistema es a través de su interfaz web.

1.  **Prepara los archivos:**
    *   Coloca los archivos de datos principal (ej. `Datos_Entrada_Secop_TVEC.csv`) dentro de la carpeta `archivos_entrada`.
    *   (Opcional) Puedes colocar archivos de `keywords` y `exclusion_words` predefinidos en la misma carpeta.

2.  **Inicia la aplicación:**
    Abre una terminal en la carpeta raíz del proyecto y ejecuta:
    ```bash
    streamlit run 0_Inicio.py
    ```

3.  **🏠 Página de Inicio: Configuración General**
    *   **Crea un Tema:** En "Gestionar Temas de Análisis", crea un nuevo tema (ej. `ciberseguridad`).
    *   **Genera el archivo Parquet:** En "Gestionar Archivo de Datos de Entrada", selecciona tu fuente (CSV, SQL, etc.) y haz clic en "Forzar Carga y Generación de Archivo Parquet". Este paso es crucial y crea la base de datos optimizada que usará el resto del sistema.
    *   **Define tus Keywords y Códigos UNSPSC:** Usa los expandibles para añadir las palabras clave y los códigos de producto/servicio que consideres relevantes.

4.  **🎛️ Panel de Control: Tu Centro de Operaciones**
    *   **Configura las Columnas:** En la tarjeta del **PASO 1**, verás dos nuevos selectores. Úsalos para indicar al sistema qué columnas contienen el **texto a analizar** y cuáles los **códigos UNSPSC**.
    *   **Genera Candidatos:** Haz clic en el botón "🚀 Generar Candidatos". El sistema leerá el archivo Parquet y usará la configuración de columnas que acabas de definir.

5.  **📝 PASO 2: Validación Humana**
    *   Esta es tu tarea principal. El sistema te presentará **únicamente los contratos nuevos**. Revisa la tabla y marca cada uno como `SI` o `NO`. Guarda tus cambios. El sistema recordará tus decisiones y no volverá a mostrarte los mismos contratos.

6.  **🧠 PASO 3: Aprender y Refinar**
    *   **Opción 1 (Recomendado):** Haz clic en "Generar Reporte de Keywords". Esto analizará el rendimiento de tus keywords y **automáticamente regenerará el dataset de entrenamiento** para el siguiente paso.
    *   **Opción 2 (Avanzado):** Haz clic en "Entrenar Modelo Experto". El sistema consolidará todo tu historial de validaciones y **re-entrenará** el modelo experto para hacerlo más inteligente.

7.  **🏆 PASO 4: Entrenar Clasificador Final**
    *   Usa todas tus validaciones para entrenar el modelo de clasificación final.

8.  **✨ PASO 5: Clasificar con Predicciones**
    *   Aplica el modelo entrenado a **toda** tu base de datos.
    *   Descarga los resultados en formato **CSV (completo)** o **Excel (solo los 'SI')**.

9.  **✨ PASO 6: Revision Activa**
    *   Revisa los resultados de la clasificacion con predicciones y genera un reporte para una segunda validacion
    *   Retoma la segunda validacion del usuario para  **calibrar el modelo**.

---

## 📁 Estructura del Proyecto

```
│
├── .streamlit/                    # Configuración de Streamlit
│   └── config.toml
├── archivos_entrada/              # Archivos de entrada organizados por tema
│   ├── ciberseguridad/            # Carpeta específica por tema
│   │   ├── Datos_Entrada_Secop.csv
│   │   ├── keywords.xlsx
│   │   └── exclusion_words.xlsx
│   ├── keywords.xlsx              # Keywords generales (opcional)
│   └── exclusion_words.xlsx      # Exclusiones generales (opcional)
├── assets/                        # Recursos multimedia
│   ├── logo.png
│   ├── banner_sistema.png
│   ├── ciclo_analisis.png
│   └── pasos_completos.png
├── pages/                         # Páginas de la interfaz web
│   ├── 0_Fuente_de_Datos.py       # Configuración de fuente de datos
│   ├── 1_Configuracion_de_Busqueda.py  # Configuración inicial
│   ├── 2_Panel_de_Control.py      # Panel principal de control
│   ├── 3_Estado_del_Proyecto.py   # Dashboard del proyecto
│   ├── 4_Historial_de_Modelos.py  # Historial de entrenamientos
│   ├── 5_Guia_de_Uso.py          # Documentación de uso
│   ├── 6_PASO_1_Generar_Candidatos.py
│   ├── 7_PASO_2_Validacion_Humana.py
│   ├── 8_PASO_3_Aprender_y_Refinar.py
│   ├── 9_PASO_4_Entrenar_Clasificador.py
│   └── 10_PASO_5_Clasificar_con_Predicciones.py
├── resultados/                    # Resultados organizados por tema
│   └── ciberseguridad/            # Carpeta específica por tema
│       ├── historial_validaciones/
│       │   └── validacion_20250722_153000.xlsx
│       ├── modelos_afinados/      # Modelos fine-tuned
│       │   └── ciberseguridad-experto-v1/
│       │       ├── config.json
│       │       ├── model.safetensors
│       │       ├── tokenizer.json
│       │       └── ...
│       ├── logs/                  # Logs de ejecución
│       │   ├── data_loader_ciberseguridad_20250722.log
│       │   ├── finetuning_ciberseguridad_20250722.log
│       │   └── ejecutar_clasificador_20250722.log
│       ├── clasificador_v1.joblib          # Modelo clasificador final
│       ├── datos_preprocesados.parquet     # Datos estandarizados
│       ├── contratos_para_revision_humana.xlsx
│       ├── finetuning_dataset.csv          # Dataset para fine-tuning
│       ├── historial_entrenamiento.csv     # Historial de entrenamientos
│       ├── revision_activa.xlsx            # Revisión activa de casos dudosos
│       ├── reporte_clasificacion.json      # Métricas del clasificador
│       ├── reporte_rendimiento_keywords.xlsx
│       ├── predicciones.csv                # Predicciones finales (CSV)
│       ├── predicciones_SI.xlsx            # Solo predicciones positivas
│       ├── hashes_validados.csv            # Control de duplicados
│       ├── embeddings_cache.pt             # Cache de embeddings
│       └── tfidf_cache.joblib              # Cache de TF-IDF
├── backups/                       # Copias de seguridad
│   └── backup_ciberseguridad_20250722_120000/
├── checkpoints/                   # Checkpoints de entrenamiento
│   └── model/
├── # --- Scripts de Procesamiento ---
├── 0_Inicio.py                    # Aplicación principal de Streamlit
├── 1_Seleccion_Candidatos.py      # Generación de candidatos
├── 2_Analisis_Feedback_Keywords.py # Análisis de rendimiento
├── 3_Generar_Datos_FineTuning.py  # Preparación datos fine-tuning
├── 4_Entrenar_Modelo_Preclasificacion.py # Fine-tuning
├── 5_Entrenamiento_Clasificador.py # Entrenamiento clasificador final
├── 6_Ejecutar_Clasificador.py     # Clasificación masiva
├── 7_Generar_Revision_Desde_Predicciones.py # Revisión activa
├── 8_Consolidar_Validaciones.py   # Consolidación de validaciones
├── 9_Consulta_datos_abiertos.py  # Consulta informacion en API SODA o Socrata de datos abiertos
├── # --- Archivos de Configuración ---
├── config.json                    # Configuración central del sistema
├── paquetes.txt                   # Dependencias Python
├── README.md                      # Documentación del proyecto
├── utils.py                       # Funciones utilitarias centralizadas
├── create_backup.py               # Sistema de copias de seguridad
└── start_app.sh                   # Script de inicio (macOS/Linux)
```

---

## Configuración Central: `config.json`

Este archivo es el cerebro de la configuración del sistema.

```json
{
  "ACTIVE_TOPIC": "ciberseguridad",
  "TOPICS": {
    "ciberseguridad": {
      "DATA_SOURCE_CONFIG": {
        "ACTIVE_SOURCE": "CSV",
        "API": {
          "BASE_URL": "",
          "API_KEY": "",
          "QUERY": ""
        },
        "CSV": {
          "FILENAME": "Datos_Entrada_Secop.xlsx"
        },
        "SQL": {
          "DB_TYPE": "mssql",
          "HOST": "servidor.empresa.com",
          "PORT": "1433",
          "DATABASE": "BodegaCCE",
          "USERNAME": "usuario",
          "PASSWORD": "contraseña",
          "DRIVER": "ODBC Driver 17 for SQL Server",
          "QUERY": "SELECT * FROM contratos WHERE fecha > '2020-01-01'"
        }
      },
      "TEXT_COLUMNS_TO_COMBINE": ["Objeto del Contrato"],
      "FILTRADO_UNSPSC": {
        "descripcion": "Códigos UNSPSC para ciberseguridad",
        "CODIGOS_DE_INTERES": [8111, 8116, 4222, 4223]
      },
      "CLASSIFIER_MODEL": "RandomForestClassifier",
      "PREDICTION_THRESHOLD": 0.85,
      "FINETUNING": {
        "NUM_EPOCHS": 2,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 3e-05,
        "WARMUP_RATIO": 0.1
      },
      "ACTIVE_LEARNING": {
        "UNCERTAINTY_THRESHOLD_LOW": 0.8,
        "UNCERTAINTY_THRESHOLD_HIGH": 0.9,
        "MAX_SAMPLES": 300
      }
    }
  }
}
```

- **`FINETUNING`**: (Recomendado) Permite ajustar los hiperparámetros del entrenamiento del modelo experto para optimizar el rendimiento.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un _issue_ para discutir cambios o reportar errores.

## Licencia

MIT





