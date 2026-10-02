# AI Log - Registro de Uso de Inteligência Artificial

As ferramentas de Inteligência Artificial foram utilizadas durante o desenvolvimento deste trabalho como apoio à programação, revisão de código, investigação de erros, estruturação de experimentos e discussão de conceitos teóricos.

As sugestões geradas pela IA foram avaliadas e, quando consideradas adequadas, implementadas e validadas. As decisões de modelagem, o design da associação de identidades, a execução dos experimentos e a interpretação dos resultados permaneceram sob responsabilidade da dupla.

Abaixo, registramos alguns episódios representativos do uso da IA durante o desenvolvimento do PA2.

## Episódio 1: Implementação da Matriz de Custos e Casamento (Partes 0 e 1)

**O problema:** A associação de identidades entre quadros exigia um casamento 1-para-1 entre deteções e tracks, maximizando a sobreposição entre as caixas. Uma implementação puramente gulosa com laços aninhados poderia produzir associações subótimas e dificultar o controle das identidades.

**Como a IA ajudou:** Consultamos a IA sobre formas de modelar a associação como um problema de *assignment*. A partir dessa discussão, avaliamos o uso de `linear_sum_assignment`, da biblioteca `scipy.optimize`, utilizando uma matriz de custos baseada em `1 - IoU`. Essa solução foi integrada ao `NaiveTracker` e realizamos testes para verificar seu comportamento.

A IA também ajudou a esclarecer que o algoritmo de *assignment* resolve apenas o casamento entre deteções e tracks em um determinado quadro.

## Episódio 2: Estrutura do Tensor e Estado Oculto da RNN (Parte 2)

**O problema:** Na Trilha A, o estado temporal do objeto é representado a partir de informações geométricas das caixas ao longo dos quadros. Foi necessário construir janelas temporais para treinamento e definir como o estado recorrente seria mantido durante a inferência.

**Como a IA ajudou:** Utilizamos a IA para verificar a dimensionalidade dos tensores de entrada do PyTorch, particularmente a organização `(batch, seq_len, features)`, e para discutir estratégias de implementação do estado recorrente.

A discussão levou à utilização de uma estrutura de tracks ativas na qual cada identidade mantém seu próprio `hidden state`. Dessa forma, estados pertencentes a objetos diferentes não são misturados e o estado de uma track pode ser propagado durante períodos sem observação direta. A estrutura final foi adaptada e implementada pela dupla de acordo com a arquitetura do projeto.

## Episódio 3: Conexão entre os Resultados do Teste de Estresse e o Modelo Temporal (Parte 5)

**O problema:** Durante o teste de estresse do detector, observamos uma degradação do IDF1 superior à degradação observada na métrica de deteção. Foi necessário interpretar esse comportamento considerando o funcionamento do modelo temporal.

**Como a IA ajudou:** Utilizamos a IA para discutir possíveis mecanismos que poderiam explicar a diferença entre a degradação da deteção e a degradação do rastreamento. A discussão ajudou a relacionar a qualidade das caixas de entrada com a atualização do estado recorrente e com a manutenção das identidades ao longo do tempo.

## Episódio 4: Avaliação de Alternativas de Implementação

**O problema:** Durante o desenvolvimento, foram consideradas diferentes estratégias para manter a identidade de um objeto durante períodos de oclusão e para atualizar o estado temporal quando não havia uma deteção confiável.

**Como a IA ajudou:** Utilizamos a IA para explorar alternativas de implementação e discutir suas vantagens e limitações. Algumas sugestões foram descartadas após serem comparadas com as restrições do PA2 e com a arquitetura adotada no projeto. A solução final foi escolhida pela dupla e validada por meio de testes sintéticos e experimentos no conjunto de dados.
