import streamlit as st 
import pickle
import torch
import time 
rutaRaiz='C:\\Master BD ENyD\\10-TFM\\'
MAX_SEQ=300 

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
    hasta 500 palabras. Una vez escrito pulse sobre el botón asociado y el sistema 
    predecirá un sentimiento político siginificando 0 izquierda y 1 derecha''')

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

    uploaded_file = st.file_uploader("Elige un archivo", type=["txt", "csv", "pdf"])
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
            for line in contenido.splitlines():
                 texto += line + "\n"
            text_area=st.text_area("Contenido del archivo:", texto, height=300)
            #resultado = predict(model,tokenizer,contenido)
            #st.text(resultado)
            spinnerWidget(model,tokenizer,text_area)
            
with open(rutaRaiz+'modelo\\roberta_model\\mpick.pkl', 'rb') as file:
    model= pickle.load(file)
    file.close()
with open(rutaRaiz+'modelo\\roberta_model\\tpick.pkl', 'rb') as file:
    tokenizer=pickle.load(file)
    file.close()
if model is None:
    st.error("No se pudo cargar el modelo. Por favor, verifica la ruta y el archivo.")
    st.stop()
printHeader(model,tokenizer)