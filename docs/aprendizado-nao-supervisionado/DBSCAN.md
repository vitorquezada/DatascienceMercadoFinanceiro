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
