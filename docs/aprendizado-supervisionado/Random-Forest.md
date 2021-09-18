---
title: "Random Forest"
---
# Conceito (O que é? Pra que serve? )
Algoritmo que usa aprendizado supervisionado para classificação ou regressão.
Combina simplicidade das árvores de decisão com a flexbilidade e aleatoriedade para uma melhor precisão.
Árvore de decisão é um conjunto de regras entre a raiz e a folha que irão determinar um resultado para uma série de perguntas. São criadas a partir de subconjuntos do dataset original. É escolhido um número de exemplos (linhas) e um número de features (colunas) de forma aleatória para criar estes subconjuntos. Após isso ele identifica quais features separam melhor os dados para ficar na raiz da árvore.

Para cada árvore de decisão teremos atributos distintos e conjuntos de dados distintos e que comporão um modelo quando combinadas. 
# Classes de Problemas com melhores resultados
Random Forest tem os melhores resultados em regressões e classificações. É bem versátil. É fácil visualizar a importância das features.
É muito prático pois os seus hiperparâmetros dão certo com praticamente todo tipo de problema e é fácil de entendê-los.
Difícilmente acontece um sobreajuste (overfitting), basta usar árvores suficientes para não ter este problema.
É ótimo para quem quer desenvolver um modelo rapidamente.

# Definição Teórica e Modelagem Matemática
É um conjunto de árvores de decisão que utiliza subconjuntos dos dados de entrada e faz treinamentos com estes subconjuntos, dessa forma o modelo utiliza as árvores de decisão que tem melhor resultado e as que são ruins tem um peso menor na hora de compor o modelo.
Com uma única árvore existe grande probabilidade de termos sobreajuste (overfitting), porém com várias árvores de decisão o peso é composto por todas e assim estamos gerando uma variancia no modelo. É um voto da maioria.

# Vantagens e Desvantagens (limitações)
Vantagens: 
- é incrivelmente versátil, pode utilizar valores binários, categóricos e numéricos.
- Precisa de pouco pré-processamento, os dados não precisam ser normalizados ou transformados.
- Pode rodar em paralelo em várias máquinas, o que resulta em processamento mais rápido.
- Ótimo com uma maior dimensionalidade (utiliza subconjuntos dos dados).
- É rápido para treinar pois usa subconjuntos.
- É robusto com outliers e dados não lineares.
- Cada árvore de decisão tem uma alta variancia e um pequeno bias, mas como é feito a média das árvores, é feito a média da variância para se ter uma variância moderada e um pequeno bias.

Desvantagens: 
- a interpretabilidade é difícil. O modelo se parece com uma caixa preta.
- Para grandes volumes de dados o tamanho das árvores pode consumir muita memória
- Ele tende ao sobreajuste (overfitting) e deve ser ajustado os hiperparâmetros


# Exemplo de uma aplicação em Python

'''

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#STEP 1: DATA READING AND UNDERSTANDING

df = pd.read_csv("other_files/images_analyzed_productivity1.csv")
print(df.head())


#Count productivity values to see the split between Good and Bad
sizes = df['Productivity'].value_counts(sort = 1)
print(sizes)

#plt.pie(sizes, autopct='%1.1f%%')
#Good to know so we know the proportion of each label


#STEP 2: DROP IRRELEVANT DATA
#In our example, Images_Analyzed reflects whether it is good analysis or bad
#so should not include it. ALso, User number is just a number and has no inflence
#on the productivity, so we can drop it.

df.drop(['Images_Analyzed'], axis=1, inplace=True)
df.drop(['User'], axis=1, inplace=True)


#STEP 3: Handle missing values, if needed
#df = df.dropna()  #Drops all rows with at least one null value. 


#STEP 4: Convert non-numeric to numeric, if needed.
#Sometimes we may have non-numeric data, for example batch name, user name, city name, etc.
#e.g. if data is in the form of YES and NO then convert to 1 and 2

df.Productivity[df.Productivity == 'Good'] = 1
df.Productivity[df.Productivity == 'Bad'] = 2
print(df.head())


#STEP 5: PREPARE THE DATA.

#Y is the data with dependent variable, this is the Productivity column
Y = df["Productivity"].values  #At this point Y is an object not of type int
#Convert Y to int
Y=Y.astype('int')

#X is data with independent variables, everything except Productivity column
# Drop label column from X as you don't want that included as one of the features
X = df.drop(labels = ["Productivity"], axis=1)  
#print(X.head())

#STEP 6: SPLIT THE DATA into TRAIN AND TEST data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)
#random_state can be any integer and it is used as a seed to randomly split dataset.
#By doing this we work with same test dataset evry time, if this is important.
#random_state=None splits dataset randomly every time

#print(X_train)

#STEP 7: Defining the model and training.

# Import the model we are using

# Import the model we are using
#RandomForestRegressor is for regression type of problems. 
#For classification we use RandomForestClassifier.
#Both yield similar results except for regressor the result is float
#and for classifier it is an integer. 
#Let us use classifier since this is a classification problem

from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 10 decision trees
model = RandomForestClassifier(n_estimators = 10, random_state = 30)
# Train the model on training data
model.fit(X_train, y_train)


#STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE

prediction_test = model.predict(X_test)
#print(y_test, prediction_test)

from sklearn import metrics
#Print the prediction accuracy
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
#Test accuracy for various test sizes and see how it gets better with more training data

#One amazing feature of Random forest is that it provides us info on feature importances
# Get numerical feature importances
#importances = list(model.feature_importances_)

#Let us print them into a nice format.

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)

'''
