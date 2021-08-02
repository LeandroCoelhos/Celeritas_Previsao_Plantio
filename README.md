# Celeritas_Previsao_Plantio

Prova 2a Fase

Chamamos de Aprendizado de Máquina o conjunto de técnicas utilizadas na Inteligência Artificial que utilizam algoritmos para aprender a realizar tarefas a partir de experiências passadas.

As técnicas de Aprendizado de Máquina podem ser divididas em uma série de categorias, das quais duas se destacam: o Aprendizado Supervisionado e o Aprendizado Não-supervisionado. 

Com o Aprendizado Supervisionado podemos resolver problemas de Classificação (separar exemplos em classes) ou Regressão (aproximar funções matemáticas complexas). Através deste tipo de técnica a máquina aprende por meio de exemplos rotulados.

Se quisermos avaliar quais alunos de uma determinada escola estão usando máscara, temos um exemplo de um problema de classificação. As classes possíveis são: aluno com máscara ou aluno sem máscara. Neste caso, cada exemplo rotulado será composto por uma foto e seu rótulo (representado pelo nome da classe a qual pertence). Tipicamente são necessários diversos exemplos rotulados para que o algoritmo possa aprender os padrões necessários para resolver uma tarefa de classificação.

Existem vários algoritmos de Classificação, dentre eles podemos citar: Árvore de Decisão (do inglês Decision Tree), k-Vizinhos Mais Próximos (do inglês k-Nearest Neighbours), Naïve Bayes, Máquina de Vetores Suporte (do inglês Support Vector Machine), etc.

Esses algoritmos podem ser implementados em diversas linguagens de programação. Muitas bibliotecas, para diversas linguagens de programação, disponibilizam implementações desses métodos de aprendizado. Além disso, para os que não tem conhecimentos de programação, existem plataformas que possibilitam a resolução de um problemas de Aprendizado de Máquina de modo bastante intuitivo.

Apenas a nível de referência, disponibilizamos abaixo alguns links de recursos que você pode utilizar para resolver esta prova:

  - Sem programação: Orange (https://orangedatamining.com/) [recomendado]
  - Programação em Python: Scikit-learn (https://scikit-learn.org/stable/)

Para utilizar o software Orange você não precisa ter conhecimento prévio em programação, já para utilizar a biblioteca Scikit-learn você precisa ter conhecimento prévio de programação na linguagem Python, a mais comumente utilizada na comunidade de IA.

Abaixo elencamos alguns conteúdos sobre a utilização do Orange e da Scikit-learn que podem ser utilizados para a construção da solução desta prova:

  -- Salvando Predições no Orange Data Mining

  -- Machine Learning sem código. Usando Orange Data Mining para criar um… | by Bruno Batista | Ensina.AI

  -- 🍊-04 PREDIÇÕES - ORANGE CANVAS

  -- Introdução ao Colab e Scikit-learn

  -- Implementando um Modelo de Classificação no Scikit-Learn*

Nesta prova iremos trabalhar com um problema de Classificação, que será detalhado abaixo.

A Inteligência Artificial tem se mostrado uma forte aliada para elevar a produtividade do agronegócio. Atualmente, além da tradicional previsão do tempo, muitos agricultores já têm acesso à pré medições de temperatura, precipitação, direção e velocidade dos ventos e demais fenômenos climáticos que podem influenciar positivamente ou negativamente nas plantações. Tendo em vista este contexto, uma fazenda de arroz da região Sul do Brasil disponibilizou dados sobre fenômenos climáticos coletados diariamente durante 7 anos, os quais devem ser utilizados para a construção de um modelo capaz de prever se os fenômenos climáticos são ou não são ideais para plantar. A planilha de treinamento fornecida tem 1.794 instâncias (linhas) e 15 características das instâncias (colunas).
<img src='https://s3-sa-east-1.amazonaws.com/datagateway-prod/images/9nJKAh1pMMo2fx5ZjqaHsEcuxlF1uqU2O_ARBcY2dNE8nAdD4BYSbJnBWO2kxAmHzQmpVGdM8W7h7i4LlaSLtBwbZBzGHOxNdOeM7iceIMsnu12mnGuQ1q7WfmMUhFx-.png'>Figura 1: Exemplo dos dados de treinamento. Fonte: Celeritas.

A Figura 1 apresenta um pedaço do conjunto de treinamento. A coluna id identifica cada uma das instâncias individualmente, apresentando valores de 1 a 1.794. A coluna target apresenta valores 1 para instâncias que representam momentos ideais para plantar, caso contrário, apresenta valores 0. As outras colunas, contendo números reais, representam dados climáticos que devem embasar a decisão sobre o plantio.

Após treinar o seu modelo, você deve realizar a predição na planilha de teste. A planilha de teste fornecida tem 599 instâncias (linhas) e 15 características das instâncias (colunas). Ela contém informações semelhantes à planilha de treinamento, mas é fornecida sem os rótulos, ou seja, se os fenômenos climáticos são ou não são ideais para plantar.
<img src='https://s3-sa-east-1.amazonaws.com/datagateway-prod/images/cqmRJmMcAn6PZdqpEPMR617mqYqsUYLc_9rFjigRiVqWjGaGlrR6oX7C_dGtPEBc0N69xs33XYMQXSxMLft9Nqs9N6zYR-xVJTZ_bjZJ19-6sa2i8bLYAvH89pj5hbaJ.png'>Figura 2: Exemplo dos dados de teste. Fonte: Celeritas.

Quando você julgar que criou um modelo competitivo envie suas predições pelo aplicativo utilizando um arquivo com a extensão .csv com exatamente 599 linhas e 2 colunas, uma linha para cada instância de teste. A primeira coluna de cada linha deve conter o Id da instância e a segunda coluna a Predição para esta instância. Este arquivo deve seguir o seguinte formato:

----------------------------------------------------------------------------------------------------------

Id, Predição

Onde:

  -- O Id representa o identificador único, um valor inteiro entre 1 e 599;
  -- , [vírgula] o separador das duas colunas;
  -- A Predição, um valor inteiro que pode ser 1 (para momentos ideais para plantar) ou 0 (para momentos não ideais para plantar).
---------------------------------------------------------------------------------------------------------

Você também deve enviar um pequeno relatório descrevendo os passos que utilizou para criar o modelo, qual software ou biblioteca foi utilizada, bem como, o porque tomou determinada decisão (arquivo com a extensão .txt). Este arquivo deve seguir o seguinte padrão:

----------------------------------------------------------------------------------------------------------

Software / Biblioteca utilizado: Orange / Scikit-learn

Passos da resolução da Prova: 1. Carregar os dados de treinamento. 2. … 3. ….

Explicação da escolha do algoritmo de Aprendizado de Máquina: Escolhi Árvore de Decisão porque …
----------------------------------------------------------------------------------------------------------

Os resultados dos participantes serão avaliados pelo desempenho de suas predições sobre o conjunto de teste, utilizando a métrica Medida-F. O critério de desempate será o tempo, ou seja, quem enviou as suas predições antes ficará melhor posicionado no ranking.

A Medida-F é a média harmônica entre as métricas de Precisão e Revocação. Em outras palavras, a Medida-F é uma métrica que avalia o desempenho de um modelo preditivo de modo a trazer um número único que indique a sua qualidade geral.

Link para download dos arquivos para análise:

Boa prova!
