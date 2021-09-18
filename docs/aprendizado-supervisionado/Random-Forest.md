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
Vantagens: * é incrivelmente versátil, pode utilizar valores binários, categóricos e numéricos.
           * Precisa de pouco pré-processamento, os dados não precisam ser normalizados ou transformados.
           * Pode rodar em paralelo em várias máquinas, o que resulta em processamento mais rápido.
           * Ótimo com uma maior dimensionalidade (utiliza subconjuntos dos dados).
           * É rápido para treinar pois usa subconjuntos.
           * É robusto com outliers e dados não lineares.
           * Cada árvore de decisão tem uma alta variancia e um pequeno bias, mas como é feito a média das árvores, é feito a média da variância para se ter uma variância moderada e um pequeno bias.
Desvantagens: *a interpretabilidade é difícil.


# Exemplo de uma aplicação em Python
