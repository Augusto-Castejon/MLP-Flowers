package br.com.augustomateus;
import java.util.Random;

public class RedeNeural {

    // Hiperparâmetros
    private int tamanhoEntrada;
    private int tamanhoOculto;
    private int tamanhoSaida;
    private double taxaAprendizagem;

    // Pesos
    private double[][] pesosEntradaOculta;
    private double[][] pesosOcultaSaida;

    // Construtor
    public RedeNeural(int tamanhoEntrada, int tamanhoOculto, int tamanhoSaida, double taxaAprendizagem) {
        this.tamanhoEntrada = tamanhoEntrada;
        this.tamanhoOculto = tamanhoOculto;
        this.tamanhoSaida = tamanhoSaida;
        this.taxaAprendizagem = taxaAprendizagem;

        pesosEntradaOculta = new double[tamanhoEntrada][tamanhoOculto];
        pesosOcultaSaida = new double[tamanhoOculto][tamanhoSaida];

        inicializarPesosAleatorios();
    }

    // Inicialização dos pesos
    private void inicializarPesosAleatorios() {
        Random random = new Random();
        for (int i = 0; i < tamanhoEntrada; i++) {
            for (int j = 0; j < tamanhoOculto; j++) {
                pesosEntradaOculta[i][j] = random.nextGaussian() * Math.sqrt(2.0 / tamanhoEntrada);
            }
        }
        for (int i = 0; i < tamanhoOculto; i++) {
            for (int j = 0; j < tamanhoSaida; j++) {
                pesosOcultaSaida[i][j] = random.nextGaussian() * Math.sqrt(2.0 / tamanhoOculto);
            }
        }
    }

    // Função de ativação Sigmoid
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // Derivada da Sigmoid
    private double derivadaSigmoid(double x) {
        return x * (1.0 - x);
    }

    // Forward propagation
    public double[] forward(double[] entradas) {
        double[] camadaOculta = new double[tamanhoOculto];
        double[] saida = new double[tamanhoSaida];

        // Cálculo da camada oculta
        for (int i = 0; i < tamanhoOculto; i++) {
            double soma = 0.0;
            for (int j = 0; j < tamanhoEntrada; j++) {
                soma += entradas[j] * pesosEntradaOculta[j][i];
            }
            camadaOculta[i] = sigmoid(soma);
        }

        // Cálculo da saída
        for (int i = 0; i < tamanhoSaida; i++) {
            double soma = 0.0;
            for (int j = 0; j < tamanhoOculto; j++) {
                soma += camadaOculta[j] * pesosOcultaSaida[j][i];
            }
            saida[i] = sigmoid(soma);
        }

        return saida;
    }

    // Função de treinamento
    public void treinar(double[] entradas, double[] saidaEsperada) {
        double[] camadaOculta = new double[tamanhoOculto];
        double[] saida = new double[tamanhoSaida];

        // Forward propagation
        for (int i = 0; i < tamanhoOculto; i++) {
            double soma = 0.0;
            for (int j = 0; j < tamanhoEntrada; j++) {
                soma += entradas[j] * pesosEntradaOculta[j][i];
            }
            camadaOculta[i] = sigmoid(soma);
        }

        for (int i = 0; i < tamanhoSaida; i++) {
            double soma = 0.0;
            for (int j = 0; j < tamanhoOculto; j++) {
                soma += camadaOculta[j] * pesosOcultaSaida[j][i];
            }
            saida[i] = sigmoid(soma);
        }

        // Backpropagation
        double[] erroSaida = new double[tamanhoSaida];
        double[] gradienteSaida = new double[tamanhoSaida];

        for (int i = 0; i < tamanhoSaida; i++) {
            erroSaida[i] = saidaEsperada[i] - saida[i];
            gradienteSaida[i] = erroSaida[i] * derivadaSigmoid(saida[i]);
        }

        double[] erroOculta = new double[tamanhoOculto];
        double[] gradienteOculta = new double[tamanhoOculto];

        for (int i = 0; i < tamanhoOculto; i++) {
            double soma = 0.0;
            for (int j = 0; j < tamanhoSaida; j++) {
                soma += gradienteSaida[j] * pesosOcultaSaida[i][j];
            }
            erroOculta[i] = soma;
            gradienteOculta[i] = erroOculta[i] * derivadaSigmoid(camadaOculta[i]);
        }

        // Ajuste dos pesos oculta-saída
        for (int i = 0; i < tamanhoOculto; i++) {
            for (int j = 0; j < tamanhoSaida; j++) {
                pesosOcultaSaida[i][j] += taxaAprendizagem * gradienteSaida[j] * camadaOculta[i];
            }
        }

        // Ajuste dos pesos entrada-oculta
        for (int i = 0; i < tamanhoEntrada; i++) {
            for (int j = 0; j < tamanhoOculto; j++) {
                pesosEntradaOculta[i][j] += taxaAprendizagem * gradienteOculta[j] * entradas[i];
            }
        }
    }

    // Cálculo do erro
    public double calcularErro(double[] saida, double[] saidaEsperada) {
        double erro = 0.0;
        for (int i = 0; i < saida.length; i++) {
            erro += Math.pow(saidaEsperada[i] - saida[i], 2);
        }
        return erro / saida.length;
    }

    // Função para treinar a rede em múltiplas épocas
    public void treinarEmLote(double[][] entradas, double[][] saidasEsperadas, int epocas) {
        for (int epoca = 0; epoca < epocas; epoca++) {
            double erroMedio = 0.0;
            for (int i = 0; i < entradas.length; i++) {
                treinar(entradas[i], saidasEsperadas[i]);
                erroMedio += calcularErro(forward(entradas[i]), saidasEsperadas[i]);
            }
            erroMedio /= entradas.length;
            System.out.println("Época " + epoca + ": Erro médio = " + erroMedio);
        }
    }

    public static void main(String[] args) {
        // Exemplo de uso
        RedeNeural rede = new RedeNeural(2, 4, 1, 0.1);

        // Dados de entrada e saída esperada para XOR
        double[][] entradas = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[][] saidasEsperadas = {
            {0},
            {1},
            {1},
            {0}
        };

        // Treinando a rede
        rede.treinarEmLote(entradas, saidasEsperadas, 10000);

        // Testando a rede
        for (double[] entrada : entradas) {
            double[] saida = rede.forward(entrada);
            System.out.println("Entrada: " + entrada[0] + ", " + entrada[1] + " -> Saída: " + saida[0]);
        }
    }
}
