---
title: "DBSCAN"
---
# Conceito (O que é? Pra que serve? )
BSCAN (Density-based spatial clustering of applications with noise) é um algoritmo de clustering (classificação) baseado em densidade, que pode ser usado para identificar clusters de qualquer forma em um conjunto de dados contendo ruídos e outliers.
Funciona para cada ponto de um cluster, a vizinhança de um determinado raio deve conter pelo menos um número mínimo de pontos.
Não é necessário definir o número de clusters.' 

# Classes de Problemas com melhores resultados
BDSCAN funciona melhor para problemas de classificação. Principalmente quando se tem ruído nos dados, é ótimo para dados que contem classes com densidades parecidas. É bom para separar areas de alta densidade de áreas de baixa densidade.

# Definição Teórica e Modelagem Matemática
É definido uma distância (raio) dos pontos no espaço e define o cluster baseado nessa distância. Caso não tenha nenhum ponto próximo a um conjunto de pontos próximos, é categorizado como um novo cluster. O próprio algoritmo define o número de clusters de acordo com a quantidade de registros.
Dependendo da forma como esse algoritmo é iniciado pode se obter clusters diferentes.

# Vantagens e Desvantagens (limitações)
Vantagens:
- Encontra padrões não lineares;
- Robusto contra outliers;
- Resultado pode ser mais consistente que o k-means pois a inicialização dos centroides não afeta tanto o algoritmo;

Desvantagens:
- Dependendo da inicialização, um ponto pode pertencer ao cluster diferente;
- Difícil encontrar um bom valor para o parâmetro da distancia.

# Exemplo de uma aplicação em Python

"""
print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
"""
