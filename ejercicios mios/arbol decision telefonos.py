from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np
import graphviz 

iris=load_iris()

X_entrenamiento,X_test,y_entrenamiento,y_test=train_test_split(iris.data,iris.target)

arbol=DecisionTreeClassifier()

arbol.fit(X_entrenamiento,y_entrenamiento)

arbol.score(X_test,y_test)

arbol.score(X_entrenamiento,y_entrenamiento)

export_graphviz(arbol,out_file='arbol.dot',class_names=iris.target_names,
    feature_names=iris.feature_names,impurity=False,filled=True)
#impurity muestra la pureza de cada nodo

with open('arbol.dot') as f:
    dot_graph=f.read()
    
graphviz.Source(dot_graph)

caract=iris.data.shape[1]
plt.barh(range(caract),arbol.feature_importances_)
plt.yticks(np.arange(caract),iris.feature_names)
plt.xlabel('importancia de las caracteristicas ')
plt.ylabel('Caracteristica')
plt.show()

arbol=DecisionTreeClassifier(max_depth=4)
arbol.fit(X_entrenamiento,y_entrenamiento)
arbol.score(X_entrenamiento,y_entrenamiento)
arbol.score(X_test,y_test)

