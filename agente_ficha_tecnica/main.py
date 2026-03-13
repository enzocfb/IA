import pandas as pd
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# 1. Configuración de Arquitectura y Logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Cargar credenciales desde el archivo .env
load_dotenv() 
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("ERROR: No se encontró la OPENAI_API_KEY en el archivo .env")

def inicializar_rag_excel(ruta_excel: str):
    """
    Lee un archivo Excel con múltiples hojas, maneja los encabezados sucios 
    y orquesta un Agente Multi-DataFrame.
    """
    logger.info(f"Iniciando lectura del archivo maestro: {ruta_excel}")
    
    try:
        # 2. Ingesta de Datos: Lectura de hojas específicas con su respectivo 'skiprows'
        # Usamos engine='openpyxl' para asegurar compatibilidad con .xlsx
        logger.info("Cargando hoja 'ODPE'...")
        df_odpe = pd.read_excel(ruta_excel, sheet_name="ODPE", skiprows=9, engine='openpyxl')
        
        logger.info("Cargando hoja 'DISTRITO'...")
        df_distritos = pd.read_excel(ruta_excel, sheet_name="DISTRITO", skiprows=10, engine='openpyxl')
        
        logger.info("Cargando hoja 'LOCALES'...")
        df_locales = pd.read_excel(ruta_excel, sheet_name="LOCALES", skiprows=1, engine='openpyxl')
        
        logger.info(f"Datos cargados exitosamente. ODPEs: {len(df_odpe)}, Distritos: {len(df_distritos)}, Locales: {len(df_locales)}")

    except FileNotFoundError:
        logger.error("No se encontró el archivo Excel. Verifica el nombre y la ruta.")
        raise
    except ValueError as e:
        logger.error(f"Error al leer las hojas del Excel. ¿Verificaste que los nombres de las pestañas sean exactos? Detalle: {e}")
        raise

    # 3. Inicializamos el LLM (El cerebro)
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.0)

    # 4. Creamos el Agente Multi-DataFrame
    # Al pasar una LISTA de dataframes [df1, df2, df3], el LLM sabe que tiene 3 tablas distintas
    logger.info("Construyendo el Agente de Datos sobre las 3 tablas...")
    agente = create_pandas_dataframe_agent(
        llm, 
        [df_odpe, df_distritos, df_locales], # <-- Aquí pasamos la lista de hojas
        verbose=True,
        agent_type="openai-tools",
        allow_dangerous_code=True 
    )
    
    return agente

# ==========================================
# EJECUCIÓN DEL CASO DE USO
# ==========================================
# ==========================================
# EJECUCIÓN DEL CASO DE USO
# ==========================================
if __name__ == "__main__":
    # 1. Obtenemos la ruta absoluta de la carpeta donde está este script (main.py)
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Unimos la carpeta con el nombre del archivo de forma segura para cualquier Sistema Operativo
    nombre_archivo = "Ficha Tecnica EG2026v06.xlsx"
    ruta_dinamica_excel = os.path.join(directorio_actual, nombre_archivo)
    
    logger.info(f"Ruta dinámica resuelta: {ruta_dinamica_excel}")
    
    # Inicializamos el sistema pasándole la ruta completa y dinámica
    agente_onpe = inicializar_rag_excel(ruta_dinamica_excel)
    
    print("\n" + "="*50)
    print("ASISTENTE DIRECTIVO ONPE - EG2026")
    print("="*50)
    
    # Prueba 1: Consulta sobre la primera hoja (ODPE)
    pregunta_1 = "¿Cuántos electores STAE y cuántas mesas STAE hay en la ODPE de PUEBLO LIBRE?"
    print(f"\nDirectivo: {pregunta_1}")
    respuesta_1 = agente_onpe.invoke(pregunta_1)
    print(f"Asistente: {respuesta_1['output']}")
    # Prueba 2: Consulta cruzando información de diferentes DataFrames
    # El LLM es lo suficientemente inteligente para buscar en df_locales o df_distritos
    pregunta_2 = "¿Cuál es el local de votación en LIMA con mayor cantidad de mesas"
    print(f"\nDirectivo: {pregunta_2}")
    respuesta_2 = agente_onpe.invoke(pregunta_2)
    print(f"Asistente: {respuesta_2['output']}")