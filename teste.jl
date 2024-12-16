using Pkg
# Pkg.add(["Flux", "CSV", "DataFrames", "Statistics"])
using CSV, DataFrames, Flux, Statistics, Random

# Carregar o dataset e definir os nomes das colunas
iris = CSV.read("iris.csv", DataFrame; header=true)
rename!(iris, [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth, :Species])

# Substituir os rótulos por valores numéricos
labels = Dict("Iris-setosa" => 1, "Iris-versicolor" => 2, "Iris-virginica" => 3)
iris.Species = [labels[label] for label in iris.Species]

# Separar os dados em inputs e outputs
X = Matrix(iris[:, 1:4])  # características (sepal_length, sepal_width, petal_length, petal_width)
y = Flux.onehotbatch(iris.Species, 1:3)  # Codificação one-hot para as saídas

# Normalização dos dados
X = (X .- mean(X, dims=1)) ./ std(X, dims=1)

# Garantir que os dados estão em Float32
X = Float32.(X)

# Divisão treino/teste
Random.seed!(42)  # Para reprodução
n_train = Int(floor(0.8 * size(X, 1)))  # Arredonda para o inteiro mais próximo
train_indices = randperm(size(X, 1))[1:n_train]
test_indices = setdiff(1:size(X, 1), train_indices)

X_train, y_train = X[train_indices, :], y[:, train_indices]
X_test, y_test = X[test_indices, :], y[:, test_indices]

# Definir o modelo
model = Chain(
    Dense(size(X, 2), 10, relu),  # Camada oculta com 10 neurônios e ativação ReLU
    Dense(10, 3),                # Camada de saída com 3 neurônios (uma para cada classe)
    softmax                      # Função de ativação para probabilidade
)

# Configurar o otimizador
opt = Flux.setup(Adam(0.01), model)

# Função de perda
loss(x, y) = Flux.crossentropy(model(x), y)

# Treinamento do modelo
for epoch in 1:100
    # Função de perda com os dados de treino
    loss_fn = () -> loss(X_train', y_train)
    
    # Calcular os gradientes
    grads = gradient(loss_fn, Flux.params(model))
    
    # Atualizar os parâmetros do modelo
    Optimisers.apply!(opt, Flux.params(model), grads)

    # Calcular e exibir a perda no conjunto de treino
    train_loss = loss(X_train', y_train)
    println("Epoch $epoch: Training Loss = $train_loss")
end

# Avaliação do modelo
y_pred = Flux.onecold(model(X_test'), 1:3)
y_actual = Flux.onecold(y_test, 1:3)

accuracy = mean(y_pred .== y_actual)
println("Test Accuracy: ", accuracy)

# Função para teste com novos dados
function test_model(new_data::Matrix)
    # Normalizar os novos dados
    new_data = (new_data .- mean(X, dims=1)) ./ std(X, dims=1)
    new_data = Float32.(new_data)  # Garantir que os dados estão em Float32

    # Previsões
    predictions = Flux.onecold(model(new_data'), 1:3)

    return predictions
end

# Exemplo de uso do teste
new_samples = [5.1 3.5 1.4 0.2; 6.2 3.4 5.4 2.3]  # Novos exemplos
predicted_classes = test_model(new_samples)
println("Predicted Classes: ", predicted_classes)
