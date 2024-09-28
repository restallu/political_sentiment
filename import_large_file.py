
import streamlit as st
import pickle
import io

def cargar_pickle_grande(uploaded_file):
    # Leer el archivo en chunks
    CHUNK_SIZE = 1024 * 1024  # 1 MB
    bytes_io = io.BytesIO()
    
    for chunk in iter(lambda: uploaded_file.read(CHUNK_SIZE), b''):
        bytes_io.write(chunk)
    
    bytes_io.seek(0)
    
    # Cargar el objeto pickle
    return pickle.load(bytes_io)

# Widget de carga de archivos
uploaded_file = st.file_uploader("Carga tu archivo pickle", type="pkl")

if uploaded_file is not None:
    try:
        # Cargar el archivo pickle
        data = cargar_pickle_grande(uploaded_file)
        
        # Mostrar información sobre los datos cargados
        st.write("Archivo cargado exitosamente")
        st.write(f"Tipo de datos: {type(data)}")
        st.write(f"Tamaño del archivo: {uploaded_file.size / (1024 * 1024):.2f} MB")
        
        # Aquí puedes procesar o mostrar 'data' según tus necesidades
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")