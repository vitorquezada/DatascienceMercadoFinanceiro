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


# Definição Teórica e Modelagem Matemática


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
