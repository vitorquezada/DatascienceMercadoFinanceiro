---
title:  "Árvore de decisão e Comitê de árvore de decisão"
---

# Árvore de decisão
## Conceito (O que é? Pra que serve?)

São um método de classificação e regressão muito utilizado pela sua facilidade de implementação, e por serem conceitualmente simples, porém ainda sim, muito poderosas.

Basicamente todos nós já utilizamos uma arvore de decisão, principalmente se desenhamos algum tipo de fluxograma (sem loop) conforme mostra a imagem abaixo.

![Árvore de decisão](./decision-tree-fluxogram.jpg)

Estes fluxogramas são bastante usados para facilitar a visualização das informações, e das condições que levam a tomar uma decisão ou outra. E com a mesma simplicidade e facilidade, podemos implementar uma árvore de decisão.

Basicamente sua estrutura consiste em nós e folhas. Um nó pode se ligar a um outro nó, ou a uma folha que são onde uma ramificação da arvore termina. Existe um nó principal, chamado nó raiz, que representa o início da árvore, ou seja, o maior nível hierarquico da árvore.

Dito isto, a árvore de decisão é nada mais que uma árvore onde os nós representam as regras de cada nó, e as folhas representam a decisão a ser tomada.

Mas basicamente a árvore de decisão por si só, não pode ser considerada machine learning, pois podemos criar ela manualmente. Para ser considerada uma técnica de machine learning, é preciso que esta aprenda automaticamente a partir de um conjunto de dados, criando as regras dos nós e as decisões nas folhas.

## Classes de Problemas com melhores resultados

Como dito, este algorítmo serve tanto para resolver problemas de classificação e de regressão. Sua aplicabilidade é variada, extendendo a qualquer tipo de problema.

## Definição Teórica e Modelagem Matemática

Basicamente a ideia geral é particinar os dados em dados menores (sub-regiões). As sub-regiões continuarão sendo divididas recursivamente até encontrar uma região pura, ou seja, que todos os dados pertencem a uma classe apenas, ou até uma condição de parada ser alcançada, que pode ser a profundidade da arvore, ou o número de folhas, ou alguma outra que fizer sentido para o problema.

Para gerar estas sub-regiões, existem vários algorítmos, sendo eles `CART`, `ID3`, e `C4.5`. Todos eles possuem o mesmo intuito, e funcionam de forma semelhante.

Para encontrar o melhor ponto de corte, devemos considerar algumas medidas de impureza, como Gini Index, Entropia ou taxa de erro. Para encontrar um ponto de corte ótimo, pode demorar muito e acaba sendo inviável para algumas classes de problemas. Portanto focamos em encontrar o ótimo local.

Para encontrar este ótimo local, testamos as possibilidades de corte com todos os valores possíveis, e utilizamos aquele que mais nos dê maior ganho de informação, que pode ser calculado da seguinte forma:

`InfoGain(R, R_e, R_d) = H(R) - (|R_e| * H(R_e) + |R_d| * H(R_d)) / |R|`

Onde H é a impureza da região, R é a região atual, R_e é a sub-região da esquerda, R_d é a sub-região da direito e \|...\| é a quantidade de exemplos da região.

Os critérios de impureza mais comuns são Entropia e o índice de Gini.

![Entropia e Gini Index](./entropia-gini-decision-tree.png)

Onde p(c\|R) é a probabilidade de um ponto da região R pertencer a classe c, que é estimada pela razão entre quantidade de pontos em R na classe c, e total de pontos em R.

## Vantagens e Desvantagens (limitações)

Vantagens:

- Conceitualmente simples
- Poderosas
- Facilidade de interpretar
- Não necessita de normalização dos dados para sua utilização
  
Desvantagens:

- Caso a profundidade seja muito grande, pode causar overfitting, ou seja, pode ficar viciada nos dados de treinamento, performando mal nos dados desconhecidos.
- É instável, pois uma pequena variação nos dados pode resultar em árvores completamente distintas.
- O algorítmo não garante a construção da melhor estrutura para os dados de treino.

## Exemplo de uma aplicação em Python

```python
# CART on the Bank Note dataset
from random import seed
from random import randrange
from csv import reader


# Load a CSV file
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)


# Test CART on Bank Note dataset
seed(1)
# load and prepare data
filename = './database-decision-tree/data_banknote_authentication.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)

# evaluate algorithm
n_folds = 5
max_depth = 5
min_size = 10
scores = evaluate_algorithm(dataset, decision_tree,
                            n_folds, max_depth, min_size)
print(f'Scores: {scores}')
print(f'Mean Accuracy: {sum(scores)/float(len(scores)):.3f}%')
```

# Comitê de árvore de decisão (Ensembles of decisions tree)

## Conceito

Para mitigar alguns problemas relacionados a construção de uma árvore de decisão simples, foi criada a técnica de Comitê de árvore de decisão, ou ensembles of decisions tree.

Os comitês são métodos que combinam várias árvores para produzir uma predição melhor. Ou seja, utiliza das vantagens de uma árvore de decisão, e ao mesmo tempo tenta mitigar as desvantagens dela quando usada sozinha.

## Classes de Problemas com melhores resultados

Como no caso da árvore de decisão, é usado tanto para classificação quanto para regressão.

## Definição Teórica e Modelagem Matemática

Uma das técnicas é chamada **Bagging**, que é usada quando o intuito é reduzir a variação da árvore de decisão. A ideia principal é criar subconjunto de dados de treinamento aleatoriamente. Assim cada subconjunto de dado é usado para treinar sua propria arvore de decisão. O resultado são várias árvores, onde na predição o resultado de todas as árvores são usadas, tornando o resultado mais robusto do que uma única árvore.

O [random forest](./Random-Forest.md) é um algorítmo extendido do bagging.

Outra técnica é a de **Boosting**. Nesta técnica, as árvores são criadas sequencialmente, e a cada sequencia o objetivo é resolver o erro da árvore anterior. 

Quando uma entrada é classificada incorretamente, seu peso é aumentado para que a próxima predição seja classificada corretamente. Após a execução de todo os dados de treinamento e sequências, a árvore que antes tinha um vícios, ou não predizia bem, passa a ter um desempenho melhor.

Ainda pode ser usado a técnica **Gradient Boosting**, que é uma extensão do método de boosting. Esta técnica usa gradiente descendente para minimizar o erro.

## Vantagens e Desvantagens (limitações)

Vantagens:

- Resolve problemas das árvores de decisão quando usadas sozinhas.
- Lida com parâmetros faltantes, e mesmo assim gera um bom resultado
- Lida bem com várias dimensões de parâmetros

Desvantagens:

- No caso do Bagging e Random Forest, a aplicação em problemas de regressão gera um resultado um pouco pior.
- No caso do boosting, ainda é propenso ao overfitting.

## Exemplo de uma aplicação em Python

```python

```