#!/bin/bash

# Navegar al directorio del script (asegura que se ejecute desde la raíz del proyecto)
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

echo "Activando entorno Conda 'base'..."
# Inicializa conda si no está en el PATH (necesario para algunos setups)
# Si ya tienes conda init en tu .bashrc o .zshrc, esta línea puede no ser necesaria
# o puede causar un error si se ejecuta dos veces.
# Si tienes problemas, puedes comentar la siguiente línea:
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate base

if [ $? -ne 0 ]; then
    echo "Error: No se pudo activar el entorno 'base'. Asegúrate de que existe y que conda está configurado correctamente."
    exit 1
fi

echo "Lanzando la aplicación Streamlit..."
streamlit run 0_Inicio.py

echo "Proceso finalizado."