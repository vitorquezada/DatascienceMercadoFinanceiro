---
title: "Processamento de Linguagem Natural"
---
# Conceito (O que é? Pra que serve? )
Processamento de linguagem natural é um ramo da inteligencia artificial que se dedica a desenvolver a capacidade da linguagem dos seres humanos.
O NLP é indispensável para permitir que a máquina compreenda o que está sendo dito e possa estruturar uma resposta, em suma a IA usa NLP para entender a linguage humana e simulá-la.

# Classes de Problemas com melhores resultados
É utilizado em chatbots, plataformas de buscas, assistentes virtuais entre outros. Ou em qualquer tipo de programa que fará uso de linguagem humana e que necessite de uma resposta de um computador.

# Definição Teórica e Modelagem Matemática
A NLP é feita para que se entenda o significado literal de cada palavra que está sendo dita, como ela se relaciona com o contexto em que está inserida e a mensagem que está transmitindo. Por isso existem os apectos de contexto, significado sintático e semântico, análise de sentimentos e interpretação de textos. 
O Modelo de NLP precisa dominar dois elementos básicos de uma interação: a entidade e a intenção.
A intenção é o elemento principal do fluxo de NLP, em um chatbot é a razão pelo qual levou um usuário a mandar a mensagem. A partir daí, no chatbot, são criados os fluxos de conversas que levam o usuário a chegar na solução do seu problema.

É muito utilizado a tokenização de palavras, em que as palavras são separadas uma a uma em frases em que se correlacionam entre si e observa-se uma relação entre elas. Esta relação é aprendida pela máquina para cada frase observada, sendo possível ainda definir um limite de palavras que podem se relacionar.
Uma outra abordagem é criar um espaço de N dimensões em que cada palavra é categorizada nesse espaço e as palavras semelhantes ficam próximas entre si. A distância entre cada uma obedece a um critério. Por exemplo as palavras 'rei' e 'rainha' estão próximas uma da outra e a uma mesma distância ( rei a uma distancia de rainha) que as palavras 'masculino' e 'feminino' que estão a mesma distância entre si de rei e rainha, porém em uma outra parte do espaço. Essa vetorização das palavras é chamada Word2Vec e é muito usada nos dias de hoje com a NLP. 

# Vantagens e Desvantagens (limitações)
O uso de NLP como um todo traz muitos avanços na linguagem. Como por exemplo analisa em larga escala documentos, manuais, emails, dados de midias sociais, tweets, reviews e mais. Permitindo que se tenha informação de forma mais assertiva e rapida que um ser humano conseguiria.
Dessa forma com um modelo de NLP bem treinado e com boa acurácia para um determinado fim é possível reduzir custos, melhorar a satisfação do usuário e entender melhor o mercado que se deseja.

A desvantagem de se usar NLP é que em muitos casos ela pode não entender algumas figuras de linguagem como a ironia. E fica limitada a um certo assunto em que foi treinada para atuar. Dificilmente uma NLP que foi treinada para atuar com artigos médicos conseguirá a mesma acurácia se for usada para um abordagem esportiva. Sendo assim seria necessário que esta passe por um treinamento de dados esportivos para que consiga funcionar com uma acurácia aceitável.

# Exemplo de uma aplicação em Python

```

import pandas as pd
import numpy as np

df = pd.read_csv('Amazon_Unlocked_Mobile.csv')
df.head()

df.dropna(inplace=True)
df[df['Rating'] != 3]
df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)
df.head(10)


df['Positively Rated'].mean()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], df['Positively Rated'], random_state = 0)

print('X_train first entry: \n\n', X_train[0])
print('\n\nX_train shape: ', X_train.shape)

# CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer().fit(X_train)
vect.get_feature_names()[::3000]

len(vect.get_feature_names())


# transform the documents in the training data to a document-term matrix
X_train_vectorized = vect.transform(X_train)
X_train_vectorized


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


from sklearn.metrics import roc_auc_score

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))

# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))

#Tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
vect = TfidfVectorizer(min_df = 5).fit(X_train)
len(vect.get_feature_names())

X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names())

sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

print('Smallest Tfidf: \n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest Tfidf: \n{}\n'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

sorted_coef_index = model.coef_[0].argsort()

print('Smallest coef: \n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest coef: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))

# These reviews are treated the same by our current model

print(model.predict(vect.transform(['Not an issue, phone is working', 
                                   'an issue, phone is not working'])))


# n-grams
# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
vect = CountVectorizer(min_df = 5, ngram_range = (1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
len(vect.get_feature_names())

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:10]))
print('Largest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:-11:-1]))

print(model.predict(vect.transform(['not an issue, phone is working',
                                   'an issue, phone is not working'])))

```
