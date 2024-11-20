#####  FRONTEND DE USUARIO DE EVALUACIÓN DEL MODELO  ###############
#####  DE ANÁLISIS DE SENTIMIENTO EN EL LENGUAJE POLÍTICO ##########
####################################################################
####################################################################
import streamlit as st 
import git
import torch
import time 
MAX_SEQ=400 
PCMIN=0.4 #Porcentaje mínimo de la seq_max para que sea analizado
import openpyxl
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from pathlib import Path
from transformers import RobertaForSequenceClassification,RobertaTokenizer
loaded_model=False
  
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#Leer excel

def get_git_root(path):

    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel") 
    return git_root
            
df2=pd.DataFrame(columns=["lr","score","texto"])
RESPUESTAS=r'./respuestas.txt'
dirRaiz= get_git_root(RESPUESTAS)
basePath=Path(dirRaiz)
df2=pd.read_excel(basePath / 'frases.xlsx')
#st.write(f'dirRaiz: {dirRaiz}')



def actStatistics(feedback, resultado):
    try:
        # Usa una ruta absoluta
        file_path = Path(os.path.abspath(os.path.dirname(__file__))) / 'respuestas.txt'
        
        with file_path.open('a') as f:
            f.write(f"{feedback},{resultado}\n")
            f.flush()
            os.fsync(f.fileno())
        
        # Verificar el contenido del archivo
        with file_path.open('r') as f:
            content = f.read()
            #st.write("Contenido del archivo:")
            st.write(content)
            #st.write(f"Ruta del archivo: {file_path}")
        
        # Verificar si el archivo existe y su tamaño
        # if file_path.exists():
            # st.write(f"El archivo existe y su tamaño es: {file_path.stat().st_size} bytes")
        # else:
            # st.write("El archivo no existe")
            
    except IOError as e:
        st.error(f'Error de E/S grabando datos: {e}')
        sys.exit(1)
        
    except Exception as e:
        st.error(f'Excepción inesperada grabando datos: {e}')
        sys.exit(1)
    
#######################################################################
# Esta y la siguiente se encargan de dibujar las graficas:
# Gráfica 1: Muesta porcentaje de aciertos y fallos globales en %
# Gráfica 2: De entre los acertados se muestran lo de cada categoria en %
##########################################################################

def plotStatistics1():
    df=pd.read_csv(basePath / 'respuestas.txt')
    df.columns=['feedback','resultado']
    # Calcular los porcentajes
    total = len(df)
    porcentaje_unos = (df['feedback'].sum() / total) * 100
    porcentaje_ceros = 100 - porcentaje_unos

    # Crear la gráfica
    fig, ax = plt.subplots(figsize=(6, 3))

    # Datos para la gráfica
    categorias = ['Predicción correcta', 'Predicción incorrecta']
    porcentajes = [porcentaje_unos, porcentaje_ceros]
    colores = ['#66b3ff', '#ff9999']

    # Crear el gráfico de barras
    barras = ax.bar(categorias, porcentajes, color=colores)

    # Añadir etiquetas de porcentaje en las barras
    for barra in barras:
        altura = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2, altura,
            f'{altura:.1f}%', ha='center', va='bottom')

    # Personalizar la gráfica
    ax.set_ylabel('Porcentaje')
    ax.set_title('Distribución del Feedback')
    ax.set_ylim(0, 100)  # Establecer el límite del eje y de 0 a 100%

    # Mostrar la gráfica en Streamlit
    st.pyplot(fig)

# Mostrar los datos numéricos
    st.write(f"Feedback Positivo: {porcentaje_unos:.1f}%")
    st.write(f"Feedback Negativo: {porcentaje_ceros:.1f}%")
    
def plotStatistics2():
    df=pd.read_csv(basePath / 'respuestas.txt')
    df.columns=['feedback','resultado']
    # Calcular los porcentajes
    #df=df[df.feedback==1]
    total_aciertos = len(df[df.feedback==1])
    total_aciertos_der=df[(df.feedback==1)&(df.resultado==1)].shape[0]
    total_errores_der=df[(df.feedback==0) & (df.resultado==1)].shape[0]
    total_aciertos_izq=total_aciertos-total_aciertos_der
    total_errores_izq=df[(df.feedback==0) & (df.resultado==0)].shape[0]
    if (total_aciertos_der+total_errores_der)!=0:
        porcentaje_aciertos_d=100*total_aciertos_der/(total_aciertos_der+total_errores_der)
    else:
        porcentaje_aciertos_d=0
        
    if (total_aciertos_izq+total_errores_izq)!=0:
        porcentaje_aciertos_i=100*total_aciertos_izq/(total_aciertos_izq+total_errores_izq)
    else:
        porcentaje_aciertos_i=0

    # Crear la gráfica
    fig, ax = plt.subplots(figsize=(6, 3))

    # Datos para la gráfica
    categorias = ['% Aciertos Derecha', '% Aciertos Izq']
    porcentajes = [porcentaje_aciertos_d, porcentaje_aciertos_i]
    colores = ['#66b3ff', '#ff9999']

    # Crear el gráfico de barras
    barras = ax.bar(categorias, porcentajes, color=colores)

    # Añadir etiquetas de porcentaje en las barras
    for barra in barras:
        altura = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2, altura,
            f'{altura:.1f}%', ha='center', va='bottom')

    # Personalizar la gráfica
    ax.set_ylabel('Porcentaje')
    ax.set_title('% Aciertos por categoría')
    ax.set_ylim(0, 100)  # Establecer el límite del eje y de 0 a 100%

    # Mostrar la gráfica en Streamlit
    st.pyplot(fig)

# Mostrar los datos numéricos
    st.write(f"Aciertos Derecha: {porcentaje_aciertos_d:.1f}%")
    st.write(f"Aciertos Izquierda: {porcentaje_aciertos_i:.1f}%")
    st.write(f"Aciertos Izquierda: {total_aciertos_izq}")
    st.write(f"Errores Izquierda: {total_errores_izq}")
    st.write(f"Aciertos Derecha: {total_aciertos_der}")
    st.write(f"Errores Derecha: {total_errores_der}")
    with (basePath / 'respuestas.txt').open('r') as f:
            print("Contenido del archivo:")
            st.write(f.read())


# ##########################################
# Recupera el texto del resultado del modelo 
# a partir del fichero frases.xlsx
############################################
 
def getResultadoTxt(df,lr='C',score=1):
    dfaux=df[(df.lr==lr) & (df.score==score)].copy()
    i=int(np.round(len(dfaux)*random.random()))
    if i>len(dfaux)-1:
        i=len(dfaux)-1
    return dfaux.iloc[i].texto

##########################################################################
# Recibe un parámetro de entrada que es una cadena de texto de tipo String 
# y devuelve una lista de cadenas de tamaño MAX_SEQ
##########################################################################

def formatContent(textoLargo):
    textoLargoWords=textoLargo.split()
    textoLargoLista=[]
    while len(textoLargoWords)>=MAX_SEQ: 
        textoLargoLista.append(' '.join(textoLargoWords[:MAX_SEQ-1])) 
        textoLargoWords=textoLargoWords[MAX_SEQ:]
    if len(textoLargoWords)>MAX_SEQ*PCMIN:
        textoLargoLista.append(' '.join(textoLargoWords))
    return textoLargoLista
 
 ################################################################
# Cuando el usuario introduce un texto se llama a esta función 
# que devuelve la predicción mientras ejecuta un spinner de espera
##################################################################   
    
def spinnerWidget(model,tokenizer,text_area):
     with st.spinner('La frase tiene un sentimiento de.... '):    
        resultado=predict(model,tokenizer,text_area)
        if resultado == 0:        
            texto1=getResultadoTxt(df,'L',999)
            texto1+='    IZQUIERDA'
        elif resultado==1: 
            texto1=getResultadoTxt(df,'R',999)
            texto1+='    DERECHA'
        else:
            texto1=getResultadoTxt(df)
            texto1+='    CENTRO'
        st.success(texto1)

##############################################################
# Ejecuta el algoritmo de prediccion
############################################################

def predict(model,tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding='max_length', truncation=True,max_length=MAX_SEQ)
    outputs = model(**inputs)
    logits = outputs.logits
    logits = logits.squeeze()
    preds = (torch.sigmoid(logits) > 0.5).float()
    #st.write('preds',preds)
    #_, preds = torch.max(logits, 1)
    return preds.item()


#########################################################################################################
# Dibujan el interface de usuario y ejecuta la logica del interface haciendo las llamadas correspondientes
###########################################################################################################
def printStaticHeader():
    st.image("Logo_UEMC.png")
    st.title('La predicción esta ok...Las gráficas de feedback están bajo corrección')
    st.title('UNIVERSIDAD EUROPEA MIGUEL DE CERVANTES')
    st.header('MASTER EN GESTIÓN Y ANÁLISIS DE GRANDES VOLÚMENES DE DATOS: BIG DATA')
    st.subheader('PROYECTO FIN DE MASTER')
    st.text('Autor: Victor Gonzalez Laria')
    st.text('Tutor: Fernando Alonso')
    #st.text(st.__version__)
    st.link_button("Read Me","https://github.com/restallu/political_sentiment/blob/main/README.md")
    st.text(
    '''    A continuación tiene un espacio para escribir un texto de hasta 3000  
    caracteres (unas 500 palabras).  Una vez escrito, pulse sobre el botón asociado 
    y el sistema predecirá un sentimiento o tendencia política de ese texto.
    El sistema evalúa el texto propuesto 'per se'. No califica de manera genérica
    al hablante o autor de ese comentario. La respuesta se muestra en tono informal.
    En el caso de que el texto sea más largo de 3000 caracteres, deberá subir
    fichero en formato txt. El sistema lo troceará, evaluará cada trozo por separado
    y decidirá cual es el sentimiento predominante del texto. Finalmente dispone
    de la opción de evaluar la respuesta que le da el sistema''')
    
def printHeader(model,tokenizer):
    printStaticHeader()
    with st.form(key='my_form'):
        text_area= st.text_area (
            label="Entre aqui el texto:",
            height=150,
            placeholder="Texto...",
            max_chars=3000)
        submitted = st.form_submit_button(label='Submit')
    if submitted:
        if text_area:
            spinnerWidget(model,tokenizer,text_area)
        else:
            text_area="Debe Introducir un texto"

    
    uploaded_file = st.file_uploader(label="Upload File", type=["txt"],label_visibility="hidden")
    
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
            if contenido:
                contenidoList=formatContent(contenido)
                derecha=0
                izquierda=0
                with st.spinner('La frase tiene un sentimiento de.... '):  
                    for contenido in contenidoList:
                        resultado=predict(model,tokenizer,contenido)   
                        if resultado == 1:
                            derecha+=1
                        elif resultado==0: 
                            izquierda+=1
                        else:
                            assert 0
                    texto+=f'Izquierda={izquierda} Derecha={derecha}'
                    if derecha+izquierda>1: #Se  divide  el  texto en bloques
                        texto="Se muestra el ultimo bloque solamente\n"+texto
                    text_area=st.text_area("Contenido del archivo:", contenido, height=300)
                    if derecha==0:
                        resultado=0
                        resultadotxt=getResultadoTxt(df2,'L',999)
                        resultadotxt+='   IZQUIERDA'
                    elif izquierda==0:
                        resultado=1
                        resultadotxt=getResultadoTxt(df2,'R',999)
                        resultadotxt+='   DERECHA'
                    else:
                        if derecha>izquierda:
                            resultado=1
                            resultadotxt=getResultadoTxt(df2,'R',int(derecha/izquierda))
                            resultadotxt+='   DERECHA'
                        elif derecha<izquierda:
                            resultado=0
                            resultadotxt=getResultadoTxt(df2,'L',int(izquierda/derecha))
                            resultadotxt+='   IZQUIERDA'
                        else:
                            resultadotxt=getResultadoTxt(df2)
                            resultadotxt+='   CENTRO'
                    st.success(resultadotxt)
           
                st.write("¿Ha sido correcta la predicción?")
                feedback=st.feedback("thumbs")
                if feedback is not None:
                    if feedback==1:
                        st.write('Gracias por su feedback. Nos complace haber acertado')
                        st.write(f'feedback {feedback} resultado {resultado}')
                        #st.write(get_git_root(RESPUESTAS))
                        actStatistics(feedback,resultado)
                        plotStatistics1()
                        plotStatistics2()
                    else:
                        feedback==0
                        st.write('Gracias por su feedback. Lamentamos haber fallado')
                        st.write(f'feedback {feedback} resultado {resultado}')
                        #st.write(get_git_root(RESPUESTAS))
                        actStatistics(feedback,resultado)
                        plotStatistics1()
                        plotStatistics2()
                    
            else:   #if contenido
                st.text('Debe subir un fichero que tenga contenido')
                
def format():
    st.markdown("""
        <style>
        header {
            padding: 10px 0;  /* Ajusta el padding superior e inferior */
            }
            .stApp {
                 margin-top: -20px;  /* Ajusta el margen superior si es necesario */
            }
        h1 {
            font-size: 24px;  /* Ajusta el tamaño del título (h1) */
        }
        h2 {
            font-size: 18px;  /* Ajusta el tamaño del encabezado (h2) */
        }
        h3 {
            font-size: 16px;  /* Ajusta el tamaño del encabezado (h3) */
        }
        body {
            width:1200px;
        }
        </style>
""", unsafe_allow_html=True)

format() 
f1 = basePath / 'ROBERTA_MP'
f2 = basePath / 'ROBERTA_TP'

# Declare modelo as a global variable at the module level


global modelo
modelo = None
loaded_model = False

def load_model():
    global modelo, loaded_model
    if not loaded_model:
        try:
            modelo = RobertaForSequenceClassification.from_pretrained(
                './ROBERTA_MP'
            )
            modelo=modelo.to('cpu')
            loaded_model = True
            #st.write('Modelo cargado')
        except Exception as e:
            st.write(f'Error  {str(e)}')
    else:
        st.write('Modelo cargado')

def load_tokenizer():
    try:
        tokenizer = RobertaTokenizer.from_pretrained('./ROBERTA_TP')
        return tokenizer
    except Exception as e:
        st.write(f'Error cargando el tokenizador: {str(e)}')
        return None

# Load the model
load_model()

# Load the tokenizer
tokenizer = load_tokenizer()

if modelo is not None and tokenizer is not None:
    modelo.eval()
    printHeader(modelo, tokenizer)
else:
    st.write("Imposible seguir por los errores de carga")