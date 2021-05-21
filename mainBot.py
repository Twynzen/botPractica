import  nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer
import numpy
import tflearn
import tensorflow
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
        
print(palabras)
print(auxY)
print(auxX)
print(tags)        
        


