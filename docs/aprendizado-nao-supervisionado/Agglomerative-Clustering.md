---
title: "Agglomerative Clustering"
---
# Conceito (O que é? Pra que serve? )
 É um método de análise de cluster que busca construir uma hierarquia de clusters. O clustering aglomerativo é o tipo mais comum de clustering hierárquico usado para agrupar objetos em clusters com base em sua similaridade. É também conhecido como AGNES ( Agglomerative Nesting ). O algoritmo começa tratando cada objeto como um cluster singleton. Em seguida, pares de clusters são mesclados sucessivamente até que todos os clusters tenham sido mesclados em um grande cluster contendo todos os objetos. O resultado é uma representação dos objetos baseada em árvore, denominada dendrograma .

# Classes de Problemas com melhores resultados
# Definição Teórica e Modelagem Matemática

Agrupamento aglomerativo começa com N grupos, cada um contendo inicialmente uma entidade e, em seguida, os dois grupos mais semelhantes se fundem em cada estágio até que haja um único grupo contendo todos os dados. Uma heurística típica para N grande é executar primeiro k-means e depois aplicar o agrupamento hierárquico aos centros do cluster estimados. Uma árvore binária chamada dendrograma representará o processo de fusão. Os grupos iniciais (objetos) estão nas folhas (na parte inferior da figura), e nós os juntamos na árvore cada vez que dois grupos são fundidos. A altura das divisões é a diferença entre os grupos que estão sendo unidos. A raiz da árvore (que está no topo) é uma categoria com todos os dados. Produzimos um agrupamento de determinado tamanho se cortarmos a árvore em qualquer altura. Além disso, existem três variantes de agrupamento aglomerativo,Murphy, 2012 ).

# Vantagens e Desvantagens (limitações)
Vantagens: 
• O agrupamento hierárquico produz uma hierarquia, ou seja, uma estrutura que é mais informativa do que o conjunto não estruturado de aglomerados planos retornado por k- means. Portanto, é mais fácil decidir
sobre o número de clusters, olhando para o dendrograma.
• Fácil de implementar

Desvantagens

• Não é possível desfazer a etapa anterior: uma vez que as instâncias foram atribuídas a um cluster, eles não podem mais ser movidos.
• Complexidade de tempo: não adequado para grandes conjuntos de dados
• As sementes iniciais têm um forte impacto nos resultados finais
• A ordem dos dados tem impacto nos resultados finais
• Muito sensível a outliers

# Exemplo de uma aplicação em Python
Para fazer um Clustering aglomerativo em Python.

Etapa 1 - Importar a biblioteca

    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
Importamos conjuntos de dados, StandardScaler, AgglomerativeClustering, pandas e seaborn que serão necessários para o conjunto de dados.

Etapa 2 - Configurando os dados
Importamos o conjunto de dados da íris embutido e armazenamos os dados em x. Traçamos um mapa de calor para correlação de recursos.
    
    iris = datasets.load_iris()
    X = iris.data; data = pd.DataFrame(X)

    cor = data.corr()
    sns.heatmap(cor, square = True); plt.show()
Etapa 3 - modelo de treinamento e previsão de clusters
Aqui nós estamos primeiro padronizando os dados por standardscaler.

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
Agora estamos usando AffinityPropagation para agrupamento com recursos:

ligação: determina qual distância usar entre os conjuntos de observação. O algoritmo irá mesclar os pares de cluster que minimizam este critério.
n_clusters: É o número de clusters que queremos ter
afinidade: neste temos que escolher entre euclidiano, l1, l2 etc.

    clt = AgglomerativeClustering(linkage="complete", affinity="euclidean", n_clusters=5)

Estamos treinando os dados usando clt.fit e imprimindo o número de clusters.

    model = clt.fit(X_std)

Finalmente, estamos prevendo os clusters.

    clusters = pd.DataFrame(model.fit_predict(X_std))
    data["Cluster"] = clusters

Etapa 4 - Visualizando o resultado

    fig = plt.figure(); ax = fig.add_subplot(111)
    scatter = ax.scatter(data[0],data[1], c=data["Cluster"],s=50)
    ax.set_title("Agglomerative Clustering")
    ax.set_xlabel("X0"); ax.set_ylabel("X1")
    plt.colorbar(scatter); plt.show()
    
Traçamos um gráfico mais detalhado que mostrará os agrupamentos de dados em cores diferentes.
