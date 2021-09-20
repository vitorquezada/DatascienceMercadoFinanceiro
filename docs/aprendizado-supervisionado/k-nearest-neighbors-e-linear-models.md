---
title: "K-Nearest Neighbors e Linear Models"
---
# Conceito (O que é? Pra que serve?)

## K-Nearest Neighbors

K-Nearest Neighbors, também conhecido como "K-ésimo vizinho mais próximo" ou apenas KNN, é um algorítmo de machine learning supervisionado que é considerado como simples e fácil de implementar.

O aspecto principal deste algoritmo está no fato de aproveitar a distribuição dos elementos, considerando que os elementos similares estão a uma distância próxima uns dos outros.

Este algoritmo é usado para resolver problemas tanto de classificação quanto de regressão.

## Linear Models

Já os Linear Models, ou Modelos lineares, também são usados para resolver problemas de classificação e regressão.

Este tipo de algorítmo consiste em gerar uma fórmula a partir de valores conhecidos, para posteriormente predizer valores desconhecidos. Eles são considerados clássicos e não oferecem uma predição tão acertiva quanto os algorítmos atuais, porém ele é relativamente rápido para ser treinado, e geralmente são de interpretação mais rápida, o que pode ser considerada uma vantagem.

Os modelos lineares em comparação com o algorítmo KNN, tem uma perda, pois no caso dos modelos lineares, a predição se baseia numa linha estreira, enquanto que o KNN não depende estritamente de uma linha de paração.

# Classes de Problemas com melhores resultados

Como o algorítmo assume que elementos similares estão próximos uns dos outros, devemos usar este algorítmo em problemas que sabemos, ou pelo menos suspeitamos, que este comportamento esteja presente.

Por se tratar de um algorítmo simples, geralmente ele é aplicado quando poucas variáveis estão envolvidas, poucos resultados e até mesmo poucos dados de exemplo. Isso porque a medida que a quantidade destas informações aumentam, a execução do algorítmo se torna mais lenta.

Apesar de resolver problemas de classificação e regressão, geralmente é usado em aplicações de recomendação de busca, que basicamente consiste em identificar elementos próximos e indica-los, ou obter os relacionados. Podemos citar como exemplo de possíveis aplicações a recomendação de filmes do netflix, artigos num blog, vídeos no youtube, dentre outros. Certamente estas empresas citadas utilizam algoritmos mais complexos que este devido a quantidade de variáveis que devem ser analisadas.

Já no caso dos modelos lineares, também são utilizados em problemas simples. Este modelo é bastante utilizado no campo da estatística. Por se tratar de um modelo simples de ser entendido e ter seus resultados interpretados, ele é bastante usado para entender a correlação entre variáveis independentes e dependentes, assim entendendo como uma variável pode influenciar mais o resultado do que outras.

# Definição Teórica e Modelagem Matemática

O algorítmo KNN usa um conceito matemático muito conhecido, que é a determinação da distância entre pontos em um gráfico. A distância Euclidiana, que é a que aprendemos no ensino fundamental, é a forma mais popular para realizar este cálculo, porém também existem outras maneiras de calcular esta distância. Por conta das várias formas de calcular, é importante analisar o problema a ser resolvido, para então determinar qual forma de cálculo é mais indicada para trazer o melhor resultado possível.

O algorítmo consiste em:
- Carregar os dados de exemplo.
- Iniciar o valor K com um valor desejado para número de vizinhos.
- Para cada dado de exemplo.
  - Calcular a distância entre o dado de exemplo e o dado atual que se deseja classificar.
  - Adicionar a distância e o indice do exemplo em uma coleção (lista) ordenada.
- Ordenar a coleção na ordem crescente de distância.
- Obter os primeiros K elementos da lista ordenada.
- Obter os rótulos dos K elementos selecionados.
- Se o problema for:
  - **Regressão**: Retorna a média dos K primeiros rótulos.
  - **Classificação**: Retorna a moda dos K primeiros rótulos.

**Obs.:** A definição do valor de `K` muitas vezes é feita baseado em multiplas execuções até encontrar um valor satisfatório que funcione tanto com os dados de exemplo, quanto com dados que não estavam no exemplo, ou seja, dados desconhecidos que partirá de um usuário por exemplo. O valor de `K` que tiver melhor performance é o valor de `K` ideal para o problema.

No caso do *modelo linear*, queremos aproximar os parâmetros para uma função ser traçada, da seguinte forma:

`y_pred = w[0] * x[0] + w[1] * x[1] + … + w[p] * x[p] + b`

A fórmula consiste basicamente em uma função composta por `x` sendo as variáveis, `w` sendo os pesos gerados pela aprendizagem e `b` uma constante também gerada na aprendizagem.

Basicamente a diferença entre os modelos lineares se dá pela forma como os parâmetros `w` e `b` serão calculados.

A forma mais comum de calculo é a regressão linear, que consiste em minimizar o erro quadrático médio entre a saída esperada e a saída gerada.

![MSE](./mse-linear-model.png)

# Vantagens e Desvantagens (limitações)

Dentre as vantagens de se usar este algorítmo, são:
- Ambos algorítmos são bastante simples e de fácil implementação.
- No caso do KNN, não há necessidade de criar um modelo, ajustar vários parâmetros, ou fazer premissas.
- O algorítmo é versátil, podendo ser usado para classificação, regressão e até para buscas no caso do KNN.

Já as desvantagens, são:
- O algorítmo fica significativamente lento com o crescimento do número de exemplos, predições, e variáveis independentes.
- Não dá uma predição tão assertiva em comparação com outros modelos.

# Exemplo de uma aplicação em Python

## K-Nearest Neighbors

```python
from collections import Counter
import math

def knn(data, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []
    
    # 3. For each example in the data
    for index, example in enumerate(data):
        # 3.1 Calculate the distance between the query example and the current
        # example from the data.
        distance = distance_fn(example[:-1], query)
        
        # 3.2 Add the distance and the index of the example to an ordered collection
        neighbor_distances_and_indices.append((distance, index))
    
    # 4. Sort the ordered collection of distances and indices from
    # smallest to largest (in ascending order) by the distances
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
    
    # 5. Pick the first K entries from the sorted collection
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    
    # 6. Get the labels of the selected K entries
    k_nearest_labels = [data[i][-1] for distance, i in k_nearest_distances_and_indices]

    # 7. If regression (choice_fn = mean), return the average of the K labels
    # 8. If classification (choice_fn = mode), return the mode of the K labels
    return k_nearest_distances_and_indices , choice_fn(k_nearest_labels)

def mean(labels):
    return sum(labels) / len(labels)

def mode(labels):
    return Counter(labels).most_common(1)[0][0]

def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)

def main():
    '''
    # Regression Data
    # 
    # Column 0: height (inches)
    # Column 1: weight (pounds)
    '''
    reg_data = [
       [65.75, 112.99],
       [71.52, 136.49],
       [69.40, 153.03],
       [68.22, 142.34],
       [67.79, 144.30],
       [68.70, 123.30],
       [69.80, 141.49],
       [70.01, 136.46],
       [67.90, 112.37],
       [66.49, 127.45],
    ]
    
    # Question:
    # Given the data we have, what's the best-guess at someone's weight if they are 60 inches tall?
    reg_query = [60]
    reg_k_nearest_neighbors, reg_prediction = knn(
        reg_data, reg_query, k=3, distance_fn=euclidean_distance, choice_fn=mean
    )
    
    '''
    # Classification Data
    # 
    # Column 0: age
    # Column 1: likes pineapple
    '''
    clf_data = [
       [22, 1],
       [23, 1],
       [21, 1],
       [18, 1],
       [19, 1],
       [25, 0],
       [27, 0],
       [29, 0],
       [31, 0],
       [45, 0],
    ]
    # Question:
    # Given the data we have, does a 33 year old like pineapples on their pizza?
    clf_query = [33]
    clf_k_nearest_neighbors, clf_prediction = knn(
        clf_data, clf_query, k=3, distance_fn=euclidean_distance, choice_fn=mode
    )

if __name__ == '__main__':
    main()
```

## Linear Models

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# variables to store mean and standard deviation for each feature
mu = []
std = []


def load_data(filename):
    df = pd.read_csv(filename, sep=",", index_col=False)
    df.columns = ["housesize", "rooms", "price"]
    data = np.array(df, dtype=float)
    #plot_data(data[:,:2], data[:, -1])
    normalize(data)
    return data[:, :2], data[:, -1]


def plot_data(x, y):
    plt.xlabel('house size')
    plt.ylabel('price')
    plt.plot(x[:, 0], y, 'bo')
    plt.show()


def normalize(data):
    for i in range(0, data.shape[1]-1):
        data[:, i] = ((data[:, i] - np.mean(data[:, i]))/np.std(data[:, i]))
        mu.append(np.mean(data[:, i]))
        std.append(np.std(data[:, i]))


def h(x, theta):
    return np.matmul(x, theta)


def cost_function(x, y, theta):
    return ((h(x, theta)-y).T@(h(x, theta)-y))/(2*y.shape[0])


def gradient_descent(x, y, theta, learning_rate=0.1, num_epochs=10):
    m = x.shape[0]
    J_all = []

    for _ in range(num_epochs):
        h_x = h(x, theta)
        cost_ = (1/m)*(x.T@(h_x - y))
        theta -= (learning_rate)*cost_
        J_all.append(cost_function(x, y, theta))

    return theta, J_all


def plot_cost(J_all, num_epochs):
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.plot(num_epochs, J_all, 'm', linewidth="5")
    plt.show()


def test(theta, x):
    x[0] = (x[0] - mu[0])/std[0]
    x[1] = (x[1] - mu[1])/std[1]

    y = theta[0] + theta[1]*x[0] + theta[2]*x[1]
    print("Price of house: ", y)


x, y = load_data("./database-linear-model/house_price_data.txt")
y = np.reshape(y, (46, 1))
x = np.hstack((np.ones((x.shape[0], 1)), x))
theta = np.zeros((x.shape[1], 1))
learning_rate = 0.1
num_epochs = 50
theta, J_all = gradient_descent(x, y, theta, learning_rate, num_epochs)
J = cost_function(x, y, theta)
print("Cost: ", J)
print("Parameters: ", theta)

# for testing and plotting cost
n_epochs = []
jplot = []
count = 0
for i in J_all:
    jplot.append(i[0][0])
    n_epochs.append(count)
    count += 1
jplot = np.array(jplot)
n_epochs = np.array(n_epochs)
#plot_cost(jplot, n_epochs)

test(theta, [1600, 3])
```