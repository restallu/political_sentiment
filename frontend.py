import streamlit as st 
import torch
import time 
MAX_SEQ=400 
PCMIN=0.4 #Porcentaje mínimo de la seq_max para que sea analizado
import openpyxl
import pandas as pd
import random
import numpy as np

#Leer excel
df=pd.read_excel('./frases.xlsx')              
df.columns=["lr","score","texto"]


def getResultadoTxt(df,lr='C',score=1):
    dfaux=df[(df.lr==lr) & (df.score==score)]
    i=int(np.floor(len(dfaux)*random.random()))
    return dfaux.iloc[i].texto

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
        resultado=predict(model,tokenizer,text_area)
        if resultado == 0:
            texto1=getResultadoTxt(df,'L',999)
        elif resultado==1: 
            texto1=getResultadoTxt(df,'R',999)
        else:
            texto1=getResultadoTxt(df)
        st.success(texto1)

def predict(model,tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding='max_length', truncation=True,max_length=MAX_SEQ)
    outputs = model(**inputs)
    logits = outputs.logits
    _, preds = torch.max(logits, 1)
    return preds.item()

def printHeader(model,tokenizer):
    
    st.title('Interface de Usuario para Text Classification')
    st.text(
    '''   A continuación tiene un espacio para escribir un texto de hasta 3000  
    caracteres nas 500 palabras).  Una vez escrito pulse sobre el botón asociado 
    y el sistema  predecirá un sentimiento político siginificando 0 izquierda y 
    1 derecha. En el caso de que el texto sea más largo suba un fichero en formato txt''')

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

    
    uploaded_file = st.file_uploader(label="", type=["txt"],label_visibility="hidden")
    
    if uploaded_file is not None:
    # Verifica el tamaño del archivo
        file_size = uploaded_file.size  # Tamaño en bytes
        max_size =  1024 * 1024 # 1 MB en bytes
        if file_size > max_size:
            st.error(f"El archivo excede el tamaño máximo permitido de 1 MB. Tamaño actual:\
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
                if derecha==0:
                    resultadotxt=getResultadoTxt(df,'L',999)
                elif izquierda==0:
                    resultadotxt=getResultadoTxt(df,'D',999)
                else:
                    if derecha>izquierda:
                        resultadotxt=getResultadoTxt(df,'D',int(derecha/izquierda))
                    elif derecha<izquierda:
                        resultadotxt=getResultadoTxt(df,'L',int(izquierda/derecha))
                    else:
                        resultadotxt=getResultadoTxt(df)
                st.success(resultadotxt)



f1='./modelorob.pth'
#modelo=torch.load(f1,map_location=torch.device('cpu'),weights_only=False)
modelo=torch.load(f1,map_location=torch.device('cpu'))
f2='tokrob.pth'
tokenizer=torch.load(f2)
modelo.eval()
printHeader(modelo,tokenizer)
