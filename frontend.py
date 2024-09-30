import streamlit as st 
import pickle
import torch
import time 
MAX_SEQ=400 
PCMIN=0.4 #Porcentaje mínimo de la seq_max para que sea analizado

def formatContent(textoLargo):
    textoLargoWords=textoLargo.split()
    textoLargoLista=[]
    while len(textoLargoWords)>=MAX_SEQ: 
        textoLargoLista.append(' '.join(textoLargoWords[:MAX_SEQ-1])) 
        textoLargoWords=textoLargoWords[MAX_SEQ:]
    if len(textoLargoWords)>MAX_SEQ*PCMIN:
        textoLargoLista.append(' '.join(textoLargoWords))
    return textoLargoLista
    
    
def spinnerWidget(model,tokenizer,text_area):
     with st.spinner('La frase tiene un sentimiento de.... '):    
        if torch.cuda.is_available():
            device = torch.device("cpu")
            model.to(device)
        else:
            device = torch.device("cpu")
        resultado=predict(model,tokenizer,text_area)
        if resultado == 0:
            texto1='Izquierda'
        elif resultado==1: 
            texto1='Derecha'
        else:
            texto1="Para ti albert rivera era el nuevo Kennedy. Pero le perdió la cabeza"
        st.success(texto1)

def predict(model,tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding='max_length', truncation=True,max_length=MAX_SEQ)
    outputs = model(**inputs)
    logits = outputs.logits
    st.text(logits)
    _, preds = torch.max(logits, 1)
    st.text(preds)
    return preds.item()

def printHeader(model,tokenizer):
    
    st.title('Interface de Usuario para Text Classification')
    st.text(
    '''    A continuación tiene un espacio para escribir un texto de hasta 3000 caracteres 
    (unas 500 palabras).  Una vez escrito pulse sobre el botón asociado y el sistema 
    predecirá un sentimiento político siginificando 0 izquierda y 1 derecha. 
    En el caso de que el texto sea más largo suba un fichero en formato txt''')

    with st.form(key='my_form'):
        #text_input = st.text_input(label='Enter some text',max_chars=600)
        text_area= st.text_area (
            label="Entre aqui el texto:",
            height=150,
            placeholder="Texto...",
            max_chars=3000)
        submitted = st.form_submit_button(label='Submit')
    if submitted:
        spinnerWidget(model,tokenizer,text_area)

    
    uploaded_file = st.file_uploader("", type=["txt", "csv", "pdf"])
    st.markdown("""
            <style>
            div[data-testid="stFileUploadDropzone"] > div > small {
            visibility: hidden;
            }

        div[data-testid="stFileUploaderDropzoneInstructions"] > div > small::before {
        visibility: visible;
        content: "Límite de 10K por archivo";
        }
        </style>
        """, unsafe_allow_html=True)
    
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
            with st.spinner('La frase tiene un sentimiento de.... '):  
                for contenido in contenidoList:
                    texto=""
                    for line in contenido.splitlines():
                        texto += line + "\n"
                    resultado=predict(model,tokenizer,texto)   
                    if resultado == 1:
                        derecha+=1
                    elif resultado==0: 
                        izquierda+=1
                    else:
                        assert 0
                texto+=f'Izquierda={izquierda} Derecha={derecha}'
                if derecha+izquierda>1: #Se  divide  el  texto en bloques
                    texto="Se muestra el ultimo bloque solamente\n"+texto
                text_area=st.text_area("Contenido del archivo:", texto, height=300)
                if derecha>izquierda:
                    resultadotxt='DERECHA'
                elif derecha<izquierda:
                    resultadotxt='IZQUIERDA'
                else:
                    resultadotxt='Para ti albert rivera era el nuevo Kennedy. Pero le perdió la cabeza'
                st.success(resultadotxt)
                

f1='./modelorob.pth'
modelo=torch.load(f1,map_location=torch.device('cpu'),weights_only=False)
f2='tokrob.pth'
tokenizer=torch.load(f2)
modelo.eval()
printHeader(modelo,tokenizer)
