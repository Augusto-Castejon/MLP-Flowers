# Rede Neural para Classificação de Flores (Iris Dataset)
 Uma rede MLP para Classificação de Flores com Iris Dataset

Este projeto implementa uma rede neural para classificar flores do conjunto de dados Iris, utilizando uma rede neural feedforward com uma camada oculta. A rede é treinada usando o algoritmo de retropropagação (backpropagation) e utiliza a função de ativação sigmoid.

## Estrutura do Projeto

- **Main.java**: Arquivo principal onde a rede neural é treinada e testada.
- **RedeNeural.java**: Implementação da rede neural com funções de inicialização de pesos, propagação direta (forward propagation) e retropropagação (backpropagation).

## Pré-requisitos

Antes de rodar o projeto, você precisará configurar o ambiente:

1. **JDK**: Certifique-se de ter o JDK (Java Development Kit) instalado. Você pode baixá-lo no [site oficial da Oracle](https://www.oracle.com/java/technologies/javase-jdk11-downloads.html) ou usar uma distribuição alternativa, como OpenJDK.

2. **IDE**: Recomendamos usar uma IDE como o [IntelliJ IDEA](https://www.jetbrains.com/idea/) ou [Eclipse](https://www.eclipse.org/) para facilitar o desenvolvimento.

## Como Rodar o Projeto

### 1. Clone o repositório

```bash
git clone https://github.com/Augusto-Castejon/MLP-Flowers.git
cd nome-do-repositorio
```

## 2. Baixe o dataset Iris
O arquivo iris.csv pode ser baixado de várias fontes online. Certifique-se de que o arquivo esteja no mesmo diretório que o seu código ou forneça o caminho correto no código.

## 3. Compile e execute o projeto
Se você estiver usando a linha de comando, execute os seguintes comandos para compilar e rodar o projeto:

```bash
javac Main.java
java br.com.augustomateus.Main
```
## 4. Entendendo a saída
A rede neural irá imprimir a previsão para cada entrada do conjunto de dados, comparando com a classe real. Ao final, será exibido o número de acertos no teste.

Exemplo de saída:
```bash
Entrada: [5.1, 3.5, 1.4, 0.2]
Saída prevista: [0.1, 0.8, 0.1]
Classe prevista: 0 | Classe real: 1

Entrada: [4.9, 3.0, 1.4, 0.2]
Saída prevista: [0.1, 0.9, 0.1]
Classe prevista: 1 | Classe real: 1

Número de acertos: 149 de 150
```

## Detalhes da Implementação
### Rede Neural
A rede neural possui:

* Entrada: 4 características das flores (comprimento e largura da sépala e pétala).
* Camada Oculta: 20 neurônios.
* Saída: 3 classes (Iris-setosa, Iris-versicolor, Iris-virginica) codificadas em one-hot.

### Funções Importantes
* Sigmoid: Função de ativação usada tanto para a camada oculta quanto para a camada de saída.
* Retropropagação (Backpropagation): Usada para ajustar os pesos da rede com base no erro entre a previsão e a saída real.

### Dataset
O conjunto de dados Iris contém 150 instâncias de flores, cada uma com 4 características:
* Comprimento da sépala
* Largura da sépala
* Comprimento da pétala
* Largura da pétala

Cada instância está associada a uma das 3 classes de flores: Iris-setosa, Iris-versicolor, e Iris-virginica.

Contribuições
Sinta-se à vontade para contribuir com melhorias, correções ou sugestões! Para isso, basta abrir uma issue ou enviar um pull request.