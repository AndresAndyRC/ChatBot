# Importamos las librerías necesarias
import random  # Para seleccionar respuestas aleatorias
import json  # Para trabajar con datos en formato JSON
import pickle  # Para cargar los objetos serializados de palabras y clases
import numpy as np  # Para trabajar con matrices y operaciones matemáticas
import nltk  # Para el procesamiento de lenguaje natural (tokenización y lematización)
from nltk.stem import WordNetLemmatizer  # Para lematizar las palabras
from keras.models import load_model  # Para cargar el modelo de red neuronal entrenado

# Inicializamos el lematizador
lemmatizer = WordNetLemmatizer()

# Cargamos los datos de las intenciones desde el archivo JSON
intents = json.loads(open('intents_spanish.json', encoding="utf-8").read())

# Cargamos las palabras clave y clases desde los archivos pickle
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Cargamos el modelo previamente entrenado
model = load_model('chatbot_model.h5')

# Función para limpiar y tokenizar una oración
def clean_up_sentence(sentence):
    """
    Tokeniza y lematiza la oración de entrada.
    Convierte las palabras a minúsculas y las reduce a su forma base (lema).
    
    Args:
    sentence (str): La oración que el usuario ingresa.
    
    Returns:
    list: Lista de palabras lematizadas.
    """
    sentence_words = nltk.word_tokenize(sentence)  # Tokeniza la oración en palabras.
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  # Lematiza las palabras.
    return sentence_words

# Función para convertir una oración en un vector "bag of words"
def bag_of_words(sentence):
    """
    Convierte una oración en un vector binario (bag of words) que indica la presencia
    o ausencia de palabras clave del vocabulario.
    
    Args:
    sentence (str): La oración que el usuario ingresa.
    
    Returns:
    np.array: Un vector binario que representa la presencia de las palabras clave en la oración.
    """
    sentence_words = clean_up_sentence(sentence)  # Limpia y tokeniza la oración.
    bag = [0]*len(words)  # Inicializa un vector de ceros del mismo tamaño que `words`.
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:  # Si la palabra está en el vocabulario, marca un 1 en el índice correspondiente.
                bag[i] = 1
    return np.array(bag)  # Retorna el vector de "bolsa de palabras".

# Función para predecir la clase (intención) de la oración
def predict_class(sentence):
    """
    Predice la clase (intención) de la oración usando el modelo de red neuronal.
    
    Args:
    sentence (str): La oración que el usuario ingresa.
    
    Returns:
    list: Una lista de intenciones predichas con su probabilidad.
    """
    bow = bag_of_words(sentence)  # Convierte la oración en un vector de "bolsa de palabras".
    res = model.predict(np.array([bow]))[0]  # Predice la clase utilizando el modelo de red neuronal.
    ERROR_THRESHOLD = 0.25  # Umbral de probabilidad para considerar una predicción.
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Filtra las predicciones con probabilidad superior al umbral.
    results.sort(key=lambda x: x[1], reverse=True)  # Ordena las predicciones por probabilidad de mayor a menor.
    
    return_list = []  # Lista de intenciones y probabilidades.
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})  # Añade la intención y su probabilidad a la lista.
    
    return return_list

# Función para obtener la respuesta correspondiente a la intención predicha
def get_response(intents_list, intents_json):
    """
    Obtiene una respuesta aleatoria basada en la intención predicha por el modelo.
    
    Args:
    intents_list (list): Lista de intenciones predichas con su probabilidad.
    intents_json (dict): Archivo JSON que contiene las posibles intenciones y respuestas.
    
    Returns:
    str: La respuesta seleccionada aleatoriamente.
    """
    tag = intents_list[0]['intent']  # Obtiene la etiqueta de la intención predicha.
    list_of_intents = intents_json['intents']  # Lista de intenciones del archivo JSON.
    
    for i in list_of_intents:
        if i['tag'] == tag:  # Compara la etiqueta predicha con las intenciones del JSON.
            result = random.choice(i['responses'])  # Elige aleatoriamente una respuesta asociada a la intención.
            break
    return result  # Retorna la respuesta seleccionada.
