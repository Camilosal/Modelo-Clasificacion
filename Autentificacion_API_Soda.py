import pandas as pd
import time
from requests.exceptions import Timeout

import os
from dotenv import load_dotenv
from sodapy import Socrata

# Carga las variables definidas en el archivo .env al entorno actual
load_dotenv()

# Obtiene las credenciales de forma segura desde las variables de entorno
usuario = os.getenv("DATOS_GOV_USER")
password = os.getenv("DATOS_GOV_PASS")
app_token = os.getenv("DATOS_GOV_TOKEN")

# Valida que todas las variables de entorno necesarias están definidas
if not all([usuario, password, app_token]):
    print("❌ Error: Asegúrate de definir las variables de entorno en un archivo .env:")
    print("   - DATOS_GOV_USER")
    print("   - DATOS_GOV_PASS")
    print("   - DATOS_GOV_TOKEN")
else:
    try:
        client = Socrata("www.datos.gov.co",
                         app_token,
                         username=usuario,
                         password=password)
        
        print("✅ Cliente Socrata inicializado exitosamente usando variables de entorno.")
        # Ahora puedes usar el 'client' para hacer consultas.

    except Exception as e:
        print(f"❌ Error al inicializar el cliente Socrata: {e}")
        