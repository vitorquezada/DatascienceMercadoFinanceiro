---
title: "Neural Networks"
---
# Conceito (O que é? Pra que serve? )

As redes neurais artificiais (RNAs) foram inspiradas na estrutura biológica do cérebro humano, fazendo-se uma analogia com o funcionamento dos neurônios. As RNAs têm por objetivo fornecer subsídios para que a ferramenta computacional consiga, com base em um conjunto de simulações conhecidas, estender tais informações para determinada situação proposta (COSTA, 2001). 

As redes neurais artificiais (ANNs) são compostas por camadas de nós, contendo uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída. Cada nó, ou neurônio artificial, se conecta a outro e tem um peso e limite associados. Se a saída de qualquer nó individual estiver acima do valor limite especificado, esse nó será ativado, enviando dados para a próxima camada da rede. Caso contrário, nenhum dado é passado para a próxima camada da rede.

# Classes de Problemas com melhores resultados

-Redes neurais convolucionais (RNCs) contêm cinco tipos de camadas: de entradas, de convolução, de agrupamento, as completamente conectadas e as de saída. Cada camada tem um propósito específico, como de resumo, conexão ou ativação. As redes neurais convolucionais popularizaram a classificação de imagens e a detecção de objetos. Entretanto, RNCs também foram aplicadas em outras áreas como previsão e processamento de linguagem natural.

-Redes neurais recorrente (RNRs) usam informações sequenciais, como dados de registro de data e hora de um sensor ou uma frase dita. Essas informações são compostas por uma sequência de termos. Diferentemente das redes neurais tradicionais, as entradas de uma rede neural recorrente não são independentes umas das outras, e os resultados para cada elemento dependem da computação dos elementos precedentes. RNRs são utilizadas na previsão e aplicação de séries temporais, análise de sentimento e outras aplicações de texto.

-Redes neurais feedforward, nas quais cada perceptron em uma camada é conectado a todo perceptron da camada seguinte. A informação é entregue de maneira antecipada de uma camada à seguinte seguindo sempre em frente. Não há loops de feedback.

-Redes neurais autoencoder são utilizadas para criar abstrações chamadas encoders, criados a partir de um conjunto estipulado de entradas. Apesar de similares às redes neurais mais tradicionais, autoencoders procuram modelar as entradas por si só e, portanto, o método é considerado não supervisionado. A premissa dos autoencoders é diminuir a sensibilidade ao que é irrelevante e aumentar ao que é. Conforme camadas são adicionadas, outras abstrações são formuladas em camadas mais altas (camadas mais próximas ao ponto onde uma camada decodificadora é introduzida). Essas abstrações podem, então, ser usadas por classificadores lineares ou não lineares.

# Definição Teórica e Modelagem Matemática
Muitos modelos são usados; definido em diferentes níveis de abstração e modelagem de diferentes aspectos dos sistemas neurais. Eles variam de modelos de comportamento de curto prazo de neurônios individuais , passando por modelos da dinâmica de circuitos neurais decorrentes das interações entre neurônios individuais, até modelos de comportamento decorrentes de módulos neurais abstratos que representam subsistemas completos. Isso inclui modelos da plasticidade de longo e curto prazo dos sistemas neurais e sua relação com o aprendizado e a memória, desde o neurônio individual até o nível do sistema.

Modelagem Matemática
um modelo de regressão linear, que pode ser entendido como uma rede neural com um único neurônio:

![image](https://user-images.githubusercontent.com/51426454/134055353-ef7a28ff-4b3f-4795-9f04-00ce106d81d1.png)

(Por simplicidade de notação, vamos omitir ϵϵ). Para adicionar mais neurônios nessa rede neural, basta então expandir a matriz de parâmetros. Além disso, vamos multiplicar a multiplicação de matrizes por mais um vetor, mantendo a consistência do output. Temos assim o modelo de uma rede neural com mais neurônios:

![image](https://user-images.githubusercontent.com/51426454/134055446-e0cb4f1f-8a05-4e9a-b017-f834847da09c.png)

É importante perceber que a matriz WW é a camada oculta da rede neural e cada coluna dessa matriz é um neurônio da camada oculta. Nós podemos pensar no vetor ww como uma camada de saída com um único neurônio, que recebe o sinal dos neurônios anteriores, pondera-os e produz o output final da rede.

A rede neural acima não é muito interessante do ponto de vista prático pois só consegue representar funções lineares. Felizmente, podemos arrumar isso facilmente, alterando o modelo da seguinte forma:

![image](https://user-images.githubusercontent.com/51426454/134055535-49baf9d6-873b-49da-bb80-edbe72b0c50c.png)

Em que ϕ é alguma função não linear diferenciável. Ela precisa ser diferenciável pois vamos treinar a rede neural com gradiente descendente, assim como fizemos num tutorial passado. O tipo mais comum de função não linear que utilizamos é a unidade linear retificada, ou ReLU:

![image](https://user-images.githubusercontent.com/51426454/134055624-9b9d3117-1867-4225-9d5d-ee3bd94f8184.png)

Formalmente, a ReLU é definida como f(x)=max(0,x). Essa função tem propriedades muito interessantes, como ser parcialmente linear, o que facilita na hora do treinamento, e ter derivadas muito simples: 0, se x<0 e 1, se x>0. (Na prática, o ponto onde a derivada não está definida é implementado como fazendo parte de alguma das regiões onde ela é bem comportada).

Quando colocamos a não linearidade, a RNA consegue representar qualquer função, dado um número suficiente de neurônios. Quanto maior o número de neurônios, maior a capacidade do modelo. É importante ressaltar também que, quando introduzimos a não linearidade na rede neural, a função custo que otimizaremos se torna não convexa e extremamente complicada de otimizar, dificultando consideravelmente o processo de treinamento.

No modelo de RNA acima, nós só utilizamos uma camada oculta, mas nada impede que utilizemos um número maior. Por exemplo, podemos construir uma rede neural artificial com duas camadas ocultas da seguinte forma:

![image](https://user-images.githubusercontent.com/51426454/134055670-d756f97e-96a4-42fe-a0cc-6c31ea7e617f.png)


# Vantagens e Desvantagens (limitações)
Vantagens da aplicação das redes neurais artificiais
a) Qualidade superior: as redes permitem análises superiores às conseguidas com té´cnicas estatísticas.
b)Competitividade : empresas que tenham conseguido redes bem elaboradas possuem maior poder de fôlego frente aos seus concorrentes, dado que essa é uma tecnologia ainda nova e pouco conhecida.
c) auto-aprendizado: não necessitam de conhecimentos de especialistas para tomar decisões; elas se baseiam unicamente nos exemplos históricos que lhes são fornecidos.
d) implementação mais rápida:o tempo necessário para se implementar uma rede é menor que o utilizado para a construção de um sistema especialista equivalente, além do menor custo envolvido.
e) imunidade a falhas
f) Capacidade de generalização
g) Imunidade a ruídos
h)Adaptabilidade
i) Demoracritização

Desvatagens:
a) treinamento demorado: O treinamento de uma rede, depende da aplicação, pode ser demorado (horas ou mesmo dias).
b) Resultados desconcertantes: as redes podem chegar a conclusões que contrariem as regras e teorias estabelecidades, bem como considerar dados irrelevantes como básicos; somente o bom senso do profissional experiente saberá tratar dois casos.
c) caixa-preta: é impossível saber o motivo que levou a rede uma coonclusão; seus critérios decisórios são encriptados, não se sabendo que valores são relevantes à tomada de uma decisão.
d) volume grande de dados
e)preparação dos dados

# Exemplo de uma aplicação em Python
