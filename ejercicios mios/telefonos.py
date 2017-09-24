from sklearn import tree
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np
import graphviz 

atributos = [[3415657634],
             [1134036714],
             [3412525252],
             [3476223399]]

etiqueta = [0,1,1,2]

clasificador = tree.DecisionTreeClassifier()
clasificador = clasificador.fit(atributos,etiqueta)

print(clasificador.predict(3476568788))

export_graphviz(clasificador,out_file='arbo.dot',filled=True)
#impurity muestra la pureza de cada nodo

with open('arbo.dot') as f:
    dot_graph=f.read()
    
graphviz.Source(dot_graph)
