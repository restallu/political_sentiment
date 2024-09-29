import streamlit as st 
import pickle
import torch
import time 
rutaRaiz='C:\\Master BD ENyD\\10-TFM\\'
url="https://drive.google.com/file/d/19bmJ0Kp5-91sEIgpyyqHjxvowjo1w6FA/view?usp=sharing"
MAX_SEQ=400 
import io
import requests

def formatContent(textoLargo):
    textoLargoWords=textoLargo.split()
    textoLargoLista=[]
    while len(textoLargoWords)>=MAX_SEQ: 
        textoLargoLista.append(' '.join(textoLargoWords[:MAX_SEQ-1])) 
        textoLargoWords=textoLargoWords[MAX_SEQ:]
    textoLargoLista.append(' '.join(textoLargoWords))
    return textoLargoLista
    
def descargar_y_cargar_pickle(url):
    response = requests.get(url)
    if response.status_code == 200:
        content = response.content
        bytes_io = io.BytesIO(content)
        return pickle.load(bytes_io)
    else:
        raise Exception(f"No se pudo descargar el archivo. Código de estado: {response.status_code}")


        
def spinnerWidget(model,tokenizer,text_area):
     with st.spinner('La frase tiene un sentimiento de.... '):    
        if torch.cuda.is_available():
            device = torch.device("cpu")
            model.to(device)
        else:
            device = torch.device("cpu")
        resultado=predict(model,tokenizer,text_area)
        if resultado == 1:
            texto1='Derecha'
        else: 
            texto1='Izquierda'
        st.success(texto1)

def predict(model,tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding='max_length', truncation=True,max_length=MAX_SEQ)
    outputs = model(**inputs)
    #outputs=model(inputs)
    logits = outputs.logits
    _, preds = torch.max(logits, 1)
    return preds.item()

def printHeader(model,tokenizer):
    
    st.title('Interface de Usuario para Text Classification')
    st.text('''A continuación tiene un espacio para escribir un texto de 
    hasta 3000 caracteres (unas 500 palabras). Una vez escrito pulse sobre el botón asociado y el sistema 
    predecirá un sentimiento político siginificando 0 izquierda y 1 derecha. En el caso de que el texto sea más largo 
    suba un fichero en formato txt''')

    # Using the "with" syntax
    with st.form(key='my_form'):
        #text_input = st.text_input(label='Enter some text',max_chars=600)
        text_area= st.text_area (
            label="Entre aqui el texto:",
            height=150,
            placeholder="Texto...",
            max_chars=3000)
        submitted = st.form_submit_button(label='Submit')
    if submitted:
        #model=loadModel(rutaRaiz+'modelo/llm_model_ft.sav')
        spinnerWidget(model,tokenizer,text_area)
    st.markdown("""
    <style>
    .uploadfile {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-family: sans-serif;
        border-radius: 0.3rem;
        cursor: pointer;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)
 

    uploaded_file = st.file_uploader("", type=["txt", "csv", "pdf"],key="txt_uploader")
    st.markdown('<label for="txt_uploader" class="uploadfile">Cargar archivo pickle</label>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
    # Verifica el tamaño del archivo
        file_size = uploaded_file.size  # Tamaño en bytes
        max_size = 10 * 1024  # 10K MB en bytes
        if file_size > max_size:
            st.error(f"El archivo excede el tamaño máximo permitido de 10K. Tamaño actual:\
                {file_size / (1024):.2f} KB")
        else:
            # Lee el contenido del archivo
            texto = ""
    # Leemos el contenido del archivo
            contenido = uploaded_file.getvalue().decode('utf-8')
            contenidoList=formatContent(contenido)
            derecha=0
            izquierda=0
            for contenido in contenidoList:
                for line in contenido.splitlines():
                    texto += line + "\n"
                resultado=predict(model,tokenizer,texto)   
                if resultado == 1:
                    derecha+=1
                else: 
                    izquierda+=1
            text_area=st.text_area("Contenido del archivo:", texto, height=300)
            if derecha>izquierda:
                st.text('Derecha')
            elif derecha<izquierda:
                st.text('Izquierda')
            else:
                st.text('Para ti albert rivera era el nuevo Kennedy. Pero le perdió la cabeza')
            #resultado = predict(model,tokenizer,contenido)
            #st.text(resultado)
            #spinnerWidget(model,tokenizer,text_area)
            
try:
    # Descargar y cargar el archivo pickle
     model = descargar_y_cargar_pickle(url)

        #st.write("Archivo cargado exitosamente")
        #st.write(f"Tipo de datos: {type(data)}")
    
    # Aquí puedes procesar o mostrar 'data' según tus necesidades
except Exception as e:
    st.error(f"Error al cargar el archivo: {str(e)}")           

with open(rutaRaiz+'modelo\\roberta_model\\tpick.pkl', 'rb') as file:
    tokenizer=pickle.load(file)
    file.close()
if model is None:
    st.error("No se pudo cargar el modelo. Por favor, verifica la ruta y el archivo.")
    st.stop()
printHeader(model,tokenizer)