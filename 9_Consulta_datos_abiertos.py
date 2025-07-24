import pandas as pd
import os
from dotenv import load_dotenv
from sodapy import Socrata

# Carga las variables definidas en el archivo .env al entorno actual
load_dotenv()

def consultar_y_guardar_soda(dataset_id: str, select_clause: str, where_clause: str, filename: str):
    """
    Realiza una consulta a la API de Socrata y guarda los resultados en un archivo CSV.

    Args:
        dataset_id (str): El identificador del conjunto de datos (ej: 'jbjy-vk9h').
        select_clause (str): La cláusula SELECT de la consulta SoQL.
        where_clause (str): La cláusula WHERE de la consulta SoQL.
        filename (str): El nombre del archivo CSV de salida.
    """
    usuario = os.getenv("DATOS_GOV_USER")
    password = os.getenv("DATOS_GOV_PASS")
    app_token = os.getenv("DATOS_GOV_TOKEN")

    if not all([usuario, password, app_token]):
        print("❌ Error: Asegúrate de definir las variables de entorno en un archivo .env:")
        print("   - DATOS_GOV_USER")
        print("   - DATOS_GOV_PASS")
        print("   - DATOS_GOV_TOKEN")
        return

    try:
        client = Socrata("www.datos.gov.co", app_token, username=usuario, password=password, timeout=600)
        print(f"✅ Cliente Socrata inicializado. Consultando el dataset '{dataset_id}'...")

        results = client.get(dataset_id, select=select_clause, where=where_clause, limit=10000000)

        if not results:
            print("No se encontraron resultados para la consulta.")
            return

        df = pd.DataFrame.from_records(results)
        print(f"Se obtuvieron {len(df)} registros.")

        # Guardar en archivo CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"✅ Datos guardados exitosamente en '{filename}'")

        print("\nPrimeras 5 filas:")
        print(df.head())

        print(f"\nColumnas disponibles: {list(df.columns)}")

    except Exception as e:
        print(f"❌ Error durante la consulta: {e}")

if __name__ == "__main__":
    # --- Parámetros de la Consulta ---
    # ID del dataset de contratos (SECOP II)
    DATASET = "jbjy-vk9h"

    # Consulta SoQL para filtrar por fecha y sin justificacion como Servicios de apoto a la gestión
    # Cláusula SELECT para especificar las columnas que quieres
    SELECT_CLAUSE = " nombre_entidad, nit_entidad, departamento, ciudad, localizaci_n, orden, sector, rama, entidad_centralizada, proceso_de_compra, id_contrato, referencia_del_contrato, estado_contrato, codigo_de_categoria_principal, descripcion_del_proceso, tipo_de_contrato, modalidad_de_contratacion, justificacion_modalidad_de, fecha_de_firma, fecha_de_inicio_del_contrato, fecha_de_fin_del_contrato, tipodocproveedor, documento_proveedor, proveedor_adjudicado, es_grupo, es_pyme, liquidaci_n, obligaci_n_ambiental, obligaciones_postconsumo, valor_del_contrato, espostconflicto, urlproceso, nombre_representante_legal, nacionalidad_representante_legal, domicilio_representante_legal, tipo_de_identificaci_n_representante_legal, identificaci_n_representante_legal, g_nero_representante_legal, ultima_actualizacion, codigo_entidad, codigo_proveedor, objeto_del_contrato, duraci_n_del_contrato "

    # Cláusula WHERE para filtrar los resultados
    WHERE_CLAUSE = "fecha_de_firma >= '2020-01-01T00:00:00' AND caseless_not_one_of(`justificacion_modalidad_de`, '', 'Servicios profesionales y apoyo a la gestión')"

    # Nombre del archivo de salida
    OUTPUT_FILENAME = "contratos_Datos Abiertos.csv"

    print("--- Iniciando consulta a Datos Abiertos ---")
    consultar_y_guardar_soda(DATASET, SELECT_CLAUSE, WHERE_CLAUSE, OUTPUT_FILENAME)
    print("--- Proceso finalizado ---")