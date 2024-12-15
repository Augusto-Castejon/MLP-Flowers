using CSV
using DataFrames  # Certifique-se de que este pacote está importado
using Flux
using Statistics

# Carregar o dataset Iris
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris = CSV.read("iris.csv", DataFrame; header=false)
names!(iris, column_names)  # Nomeando as colunas corretamente

# Pré-processamento dos dados
# Convertendo a coluna 'species' para uma variável categórica
iris.species = categorical(iris.species)

# Convertendo os dados para arrays
X = Matrix(iris[:, 1:4])  # características
y = onehotbatch(iris.species)  # rótulos (one-hot encoding)

# Dividir os dados em treino e teste
n = size(X, 1)
train_size = Int(n * 0.8)
X_train, X_test = X[1:train_size, :], X[train_size+1:end, :]
y_train, y_test = y[:, 1:train_size], y[:, train_size+1:end]

# Definir a arquitetura da rede neural MLP
model = Chain(
    Dense(4, 10, relu),  # camada oculta com 10 neurônios e ReLU
    Dense(10, 3),        # camada de saída com 3 neurônios (uma para cada classe)
    softmax              # função de ativação softmax para classificação
)

# Definir a função de perda (entropia cruzada)
loss(x, y) = crossentropy(model(x), y)

# Definir o otimizador (Stochastic Gradient Descent)
opt = ADAM()

# Função de treinamento
function train!(X_train, y_train, model, opt, epochs=100)
    for epoch in 1:epochs
        Flux.train!(loss, params(model), [(X_train', y_train)], opt)
        if epoch % 10 == 0
            println("Epoch $epoch: $(loss(X_train', y_train))")
        end
    end
end

# Treinando o modelo
train!(X_train, y_train, model, opt, epochs=100)

# Avaliar o modelo
function accuracy(X_test, y_test, model)
    y_pred = model(X_test')
    y_pred_class = argmax.(eachcol(y_pred))
    y_true_class = argmax.(eachcol(y_test))
    return mean(y_pred_class .== y_true_class)
end

# Acurácia do modelo no conjunto de teste
println("Acurácia: ", accuracy(X_test, y_test, model))
