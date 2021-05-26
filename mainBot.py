import  nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
from tensorflow.python.framework import ops
import json
import random
import pickle

# nltk.download('punkt')

with open("contenido.json") as archivo:
    datos = json.load(archivo)
    
palabras=[]
tags=[]
auxX=[]
auxY=[]

for contenido in datos["contenido"]:
    for patrones in contenido["patrones"]:
        #Esta función de word_tokenize() separa como split y reconoce caracteres especiales
        auxPalabra = nltk.word_tokenize(patrones)
        palabras.extend(auxPalabra)
        auxX.append(auxPalabra)
        auxY.append(contenido["tag"])
        
        if contenido["tag"] not in tags:
            tags.append(contenido["tag"])
        
palabras = [stemmer.stem(w.lower()) for w in palabras if w!="?"]
palabras = sorted(list(set(palabras)))     
tags = sorted(tags)

entrenamiento = []
salida = []
salidVacia = [0 for _ in range(len(tags))]

for x, documento in enumerate(auxX):
    cubeta=[]
    auxPalabra = [stemmer.stem(w.lower()) for w in documento]
    for w in palabras:
        if w in auxPalabra:
            cubeta.append(1)
        else:
            cubeta.append(0)
    filaSalida = salidVacia[:]
    filaSalida[tags.index(auxY[x])] = 1
    entrenamiento.append(cubeta)
    salida.append(filaSalida)

entrenamiento = numpy.array(entrenamiento)
salida = numpy.array(salida)

ops.reset_default_graph()
red = tflearn.input_data(shape=[None,len(entrenamiento[0])])
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, len(salida[0]),activation="softmax")
red = tflearn.regression(red) 

modelo = tflearn.DNN(red)
modelo.fit(entrenamiento, salida, n_epoch = 1000, batch_size = 10, show_metric = True)
modelo.save("modelo.tflearn")            
    
        


