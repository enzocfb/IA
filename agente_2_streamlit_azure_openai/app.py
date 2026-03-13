import streamlit as st
import os
import urllib
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

st.set_page_config(page_title="Asistente GOECOR", page_icon="📊")
st.title("📊 Asistente de Datos - GOECOR (ONPE)")

# 1. Cargar credenciales desde Secrets
os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"]

server = st.secrets["FABRIC_SERVER"]
database = st.secrets["FABRIC_DATABASE"]
username = st.secrets["FABRIC_USER"]
password = st.secrets["FABRIC_PASSWORD"]


# 2. Configurar la conexión SQL a Microsoft Fabric
@st.cache_resource
def conectar_bd():
    # Fabric requiere Autenticación de Active Directory
    driver = "{ODBC Driver 18 for SQL Server}"
    params = urllib.parse.quote_plus(
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={server},1433;"
        f"DATABASE={database};"
        f"UID={username};"
        f"Authentication=ActiveDirectoryInteractive;" # <-- ESTE ES EL CAMBIO MÁGICO
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
    )

    conexion_str = f"mssql+pyodbc:///?odbc_connect={params}"
    return SQLDatabase.from_uri(conexion_str)

# 3. Inicializar el Modelo y el Agente
@st.cache_resource
def iniciar_agente():
    db = conectar_bd()
    llm = AzureChatOpenAI(
        azure_deployment="modelo-gpt4", # Asegúrate que sea el nombre exacto
        api_version="2024-02-15-preview",
        temperature=0.0
    )
    # Crear el agente experto en SQL
    return create_sql_agent(llm=llm, db=db, verbose=True)

try:
    agente = iniciar_agente()
except Exception as e:
    st.error(f"Error al conectar con Fabric: {e}")
    st.stop()

# 4. Interfaz del Chat
if "historial" not in st.session_state:
    st.session_state.historial = []

for rol, texto in st.session_state.historial:
    with st.chat_message(rol):
        st.write(texto)

pregunta_usuario = st.chat_input("Ej: ¿Cuál es el total de capacitados en la ODPE?")

if pregunta_usuario:
    with st.chat_message("user"):
        st.write(pregunta_usuario)
    st.session_state.historial.append(("user", pregunta_usuario))
    
    with st.chat_message("assistant"):
        with st.spinner("Consultando la base de datos de la ONPE..."):
            try:
                # El agente traduce a SQL, consulta y responde
                respuesta = agente.invoke({"input": pregunta_usuario})
                texto_respuesta = respuesta["output"]
                st.write(texto_respuesta)
                st.session_state.historial.append(("assistant", texto_respuesta))
            except Exception as e:
                st.error("Hubo un problema procesando la consulta con los datos.")