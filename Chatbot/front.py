#pip3 install streamlit instala la libreria
#streamlit run front.py para correr el programa

#Importamos la libreria
import streamlit as st

# importamos la ia del chatbot
from chatbot import predict_class, get_response, intents

#Titulo de la pagina
st.title("😁 Asistente Virtual")

#variables de estado

if "messages" not in st.session_state:
    st.session_state.messages = [] #almacena los mensajes
if "first_message" not in st.session_state:
    st.session_state.first_message = True #verifica si es el primer mensaje


#funcion para recorrer el array de mensajes y mirar el historico 
#muestra el historico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#comprobar si es la primera ejecucion del codigo de ser true se muestra mensaje de bienvenida

if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hola, soy tu asistente virtual, ¿En qué puedo ayudarte?")

    st.session_state.messages.append({"role": "assistant", "content": "Hola, soy tu asistente virtual, ¿En qué puedo ayudarte?"})
    st.session_state.first_message = False

# creacion de prompt para el usuario    

if prompt := st.chat_input("Escribe un mensaje..."):

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # implementacion de la ia del chatbot
    ints = predict_class(prompt)
    res = get_response(ints, intents)

    with st. chat_message("assistant"):
        st.markdown(res)
    st.session_state.messages.append({"role": "assistant", "content": res})    