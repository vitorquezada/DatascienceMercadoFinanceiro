---
title: "K-Means Clustering"
---
# Conceito (O que é? Pra que serve? )
 É um algoritmo de aprendizado não supervisionado (ou seja, que não precisa de inputs de confirmação externos) que avalia e clusteriza os dados de acordo com suas características, como por exemplo:
lojas/centro logistico
clientes/produtos ou serviços semelhantes
clientes/características semelhantes
séries/gênero da série ou faixa etaria
usuarios de uma rede social/usuario influenciador
paciente/sintoma ou característica semelhante
Por exemplo, se eu tenho uma rede de lojas com abrangência nacional, qual seria os melhores lugares para construir os centros logísticos de abastecimento?
Podemos começar a responder isso com K-means.
 
# Classes de Problemas com melhores resultados

Quantização vetorial
K-means se origina do processamento de sinal, mas também é usado para quantização vetorial. Por exemplo, a quantização de cores é a tarefa de reduzir a paleta de cores de uma imagem a um número fixo de cores k . O algoritmo k-means pode ser facilmente usado para essa tarefa.

Psicologia e medicina
Uma doença ou condição freqüentemente tem uma série de variações e a análise de agrupamento pode ser usada para identificar essas diferentes subcategorias. Por exemplo, o agrupamento foi usado para identificar diferentes tipos de depressão. A análise de agrupamento também pode ser usada para detectar padrões na distribuição espacial ou temporal de uma doença.

Sistemas de Recomendação
O armazenamento em cluster também pode ser usado em mecanismos de recomendação . No caso de recomendar filmes a alguém, você pode ver os filmes apreciados por um usuário e então usar o agrupamento para encontrar filmes semelhantes.

Clustering de documentos
Esta é outra aplicação comum de clustering. Digamos que você tenha vários documentos e precise agrupar documentos semelhantes. O clustering nos ajuda a agrupar esses documentos de forma que documentos semelhantes fiquem nos mesmos clusters.

# Definição Teórica e Modelagem Matemática
O algoritmo de agrupamento k-means
O agrupamento K-means é uma técnica de agrupamento parcial baseada em protótipo que tenta encontrar um número de clusters (k) especificado pelo usuário, que são representados por seus centróides.

*Procedimento*

Primeiro, escolhemos k centróides iniciais, onde k é um parâmetro especificado pelo usuário; ou seja, o número de clusters desejado. Cada ponto é então atribuído ao centróide mais próximo, e cada coleção de pontos atribuídos a um centróide é chamada de cluster. O centróide de cada cluster é então atualizado com base nos pontos atribuídos ao cluster. Repetimos as etapas de atribuição e atualização até que nenhum ponto mude os clusters, ou similarmente, até que os centróides permaneçam os mesmos.


![image](https://user-images.githubusercontent.com/51426454/135013345-c1157e96-1f09-4ffb-a43a-64ff4668a995.png)

*Medidas de Proximidade*
Para clustering, precisamos definir uma medida de proximidade para dois pontos de dados. A proximidade aqui significa o quão semelhantes / diferentes as amostras são em relação umas às outras.
A medida de similaridade é grande se os recursos são semelhantes.
A medida de dissimilaridade é pequena se os recursos são semelhantes.

*Dados no Espaço Euclidiano*
Considere os dados cuja medida de proximidade é a distância euclidiana . Para nossa função objetivo, que mede a qualidade de um agrupamento, usamos a soma do erro quadrático (SSE) , que também é conhecido como dispersão .
Em outras palavras, calculamos o erro de cada ponto de dados, ou seja, sua distância euclidiana até o centróide mais próximo, e então calculamos a soma total dos erros quadrados. Dados dois conjuntos diferentes de clusters que são produzidos por duas execuções diferentes de K-médias, preferimos aquele com o menor erro quadrático, pois isso significa que os protótipos (centróides) deste cluster são uma melhor representação dos pontos em seu cluster .

![image](https://user-images.githubusercontent.com/51426454/135013449-dec3cc1d-b3bc-4ab8-9518-c4329b9140e4.png)

*Dados do Documento*
Para ilustrar que K-means não se restringe aos dados no espaço euclidiano, consideramos os dados do documento e a medida de similaridade do cosseno:
![image](https://user-images.githubusercontent.com/51426454/135013489-50e6b0a8-9a28-4f1e-a272-d7c8e4643748.png)

*Implementação no scikit-learn*
São necessárias apenas quatro linhas para aplicar o algoritmo em Python com sklearn: importe o classificador, crie uma instância, ajuste os dados no conjunto de treinamento e preveja resultados para o conjunto de teste:

![image](https://user-images.githubusercontent.com/51426454/135013540-ccfb53a1-0a81-484a-9a70-93d7cb5495a7.png)



# Vantagens e Desvantagens (limitações)
Vantagens
Relativamente simples de implementar.
Escala para grandes conjuntos de dados.
Garante a convergência.
Pode inicializar a quente as posições dos centróides.
Adapta-se facilmente a novos exemplos.
Generaliza para clusters de diferentes formas e tamanhos, como clusters elípticos.

Desvantagens
Clustering outliers.
Agrupando dados de tamanhos e densidades variados.
Dimensionamento com número de dimensões.
Sendo dependente de valores iniciais.
Escolhendo k-manualmente.

# Exemplo de uma aplicação em Python

Exemplo: a rede de lojas Bruno tem 19 lojas em algumas das principais cidades do país. A empresa pensa em construir 3 centros logísticos para abastecer as lojas. Mas, qual seria a posição ótima para cada um desses três hubs, considerando apenas a posição (coordenadas geográficas) das lojas?
Abaixo, plotamos no gráfico a representação em coordenadas de cada uma das 19 cidades onde a rede possui filiais.

dataset = np.array(
#matriz com as coordenadas geográficas de cada loja
[[-25, -46], #são paulo
[-22, -43], #rio de janeiro
[-25, -49], #curitiba
[-30, -51], #porto alegre
[-19, -43], #belo horizonte
[-15, -47], #brasilia
[-12, -38], #salvador
[-8, -34], #recife
[-16, -49], #goiania
[-3, -60], #manaus
[-22, -47], #campinas
[-3, -38], #fortaleza
[-21, -47], #ribeirão preto
[-23, -51], #maringa
[-27, -48], #florianópolis
[-21, -43], #juiz de fora
[-1, -48], #belém
[-10, -67], #rio branco
[-8, -63] #porto velho])
plt.scatter(dataset[:,1], dataset[:,0]) #posicionamento dos eixos x e y
plt.xlim(-75, -30) #range do eixo x
plt.ylim(-50, 10) #range do eixo y
plt.grid() #função que desenha a grade no nosso gráfico

O ponto mais ao sul no gráfico representa Porto Alegre, o ponto mais à Oeste representa Rio Branco, o ponto mais à leste representa Recife e o ponto mais ao norte representa Belém.

![image](https://user-images.githubusercontent.com/51426454/133943505-6b6bf555-b18a-4079-b389-cd82e4a368a8.png)


Vamos utilizar o algoritmo KMeans, do pacote Scikit-Learn para agrupar (clusterisar) as nossas filiais em 3 grupos. Cada grupo será servido por um centro logístico, que será representado por um centróide.

kmeans = KMeans(n_clusters = 3, #numero de clusters
init = 'k-means++', n_init = 10, #algoritmo que define a posição dos clusters de maneira mais assertiva
max_iter = 300) #numero máximo de iterações
pred_y = kmeans.fit_predict(dataset)
plt.scatter(dataset[:,1], dataset[:,0], c = pred_y) #posicionamento dos eixos x e y
plt.xlim(-75, -30) #range do eixo x
plt.ylim(-50, 10) #range do eixo y
plt.grid() #função que desenha a grade no nosso gráfico
plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,0], s = 70, c = 'red') #posição de cada centroide no gráfico
plt.show()

![image](https://user-images.githubusercontent.com/51426454/133943473-ccd46d95-57b7-4a6a-957d-47509a76e26a.png)

Nossa clusterização apontou três posições para os nossos centros logísticos. Vamos ver onde, aproximadamende, eles ficariam?
[-7, -63.33333333] — Humaitá/AM
[-6, -39.5] — Acopiara/CE
[-22.16666667, -47] — Mogi Guaçu/SP

Com certeza não faria sentido abrir um galpão em Humaitá/AM para abastecer apenas 3 lojas e outro em Acopiara/CE, para abastecer apenas 4 filias. Aqui entra a necessidade de completude do nosso dataset. Quanto mais dados, mais assertiva seria a nossa resposta.
