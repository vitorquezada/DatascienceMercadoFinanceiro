---
title: "PCA e AutoEncoder (Reducao de Dimensionalidade)"
---
# Conceito (O que é? Pra que serve? )
PCA
O PCA aprende essencialmente uma transformação linear que projeta os dados em outro espaço, onde os vetores de projeções são definidos pela variação dos dados. Ao restringir a dimensionalidade a um certo número de componentes que respondem pela maior parte da variância do conjunto de dados, podemos alcançar a redução da dimensionalidade.
Autoencoders
Autoencoders são redes neurais que podem ser usadas para reduzir os dados em um espaço latente de baixa dimensão, empilhando várias transformações não lineares (camadas). Eles têm uma arquitetura codificador-decodificador. O codificador mapeia a entrada para o espaço latente e o decodificador reconstrói a entrada. Eles são treinados usando retropropagação para uma reconstrução precisa da entrada. No espaço latente tem dimensões menores do que a entrada, autoencoders podem ser usados ​​para redução de dimensionalidade. Por intuição, essas variáveis ​​latentes de baixa dimensão devem codificar as características mais importantes da entrada, uma vez que são capazes de reconstruí-la.

Comparação
O PCA é essencialmente uma transformação linear, mas os codificadores automáticos são capazes de modelar funções não lineares complexas.
Os recursos do PCA são totalmente linearmente não correlacionados entre si, uma vez que os recursos são projeções na base ortogonal. Mas os recursos auto-codificados podem ter correlações, uma vez que são apenas treinados para uma reconstrução precisa.
O PCA é mais rápido e computacionalmente mais barato do que os codificadores automáticos.
Um autoencoder de camada única com função de ativação linear é muito semelhante ao PCA.
O Autoencoder está sujeito a sobreajuste devido ao grande número de parâmetros. (embora a regularização e o design cuidadoso possam evitar isso)

# Classes de Problemas com melhores resultados
# Definição Teórica e Modelagem Matemática
# Vantagens e Desvantagens (limitações)
Vantagens:
- Reduzir o custo computacional envolvido no processamento dos dados.
- Eliminar redundâncias nas informações disponíveis.
- Possibilitar a visualização dos dados.
# Exemplo de uma aplicação em Python
