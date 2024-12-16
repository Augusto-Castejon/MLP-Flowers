package br.com.augustomateus;

import java.io.*;
import java.util.*;

public class Main {

    public static void main(String[] args) {
        try {
            // 1. Carregar o dataset Iris
            List<double[]> entradas = new ArrayList<>();
            List<double[]> saidas = new ArrayList<>();
            carregarDataset("iris.csv", entradas, saidas);

            // 2. Criar a rede neural (4 entradas, 15 neurônios ocultos, 3 saídas)
            RedeNeural rede = new RedeNeural(4, 20, 3, 0.1);

            // 3. Treinar a rede
            int epocas = 10000;
            for (int i = 0; i < epocas; i++) {
                for (int j = 0; j < entradas.size(); j++) {
                    rede.treinar(entradas.get(j), saidas.get(j));
                }
            }

            // 4. Testar a rede
            int acertos = 0;
            for (int i = 0; i < entradas.size(); i++) {
                double[] resultado = rede.forward(entradas.get(i));
                int classePrevista = argMax(resultado);
                int classeReal = argMax(saidas.get(i));

                System.out.println("Entrada: " + Arrays.toString(entradas.get(i)));
                System.out.println("Saída prevista: " + Arrays.toString(resultado));
                System.out.println("Classe prevista: " + classePrevista + " | Classe real: " + classeReal);

                if (classePrevista == classeReal) {
                    acertos++;
                }
            }

            System.out.println("Número de acertos: " + acertos + " de " + entradas.size());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Método para carregar o dataset Iris
    private static void carregarDataset(String caminhoArquivo, List<double[]> entradas, List<double[]> saidas) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(caminhoArquivo));
        String linha;

        // Mapear classes para índices
        Map<String, Integer> mapaClasses = new HashMap<>();
        mapaClasses.put("Iris-setosa", 0);
        mapaClasses.put("Iris-versicolor", 1);
        mapaClasses.put("Iris-virginica", 2);

        while ((linha = br.readLine()) != null) {
            String[] valores = linha.split(",");
            double[] entrada = new double[4]; // 4 características do Iris
            double[] saida = new double[3];  // 3 classes (one-hot encoding)

            for (int i = 0; i < 4; i++) {
                entrada[i] = Double.parseDouble(valores[i]);
            }

            // Obter o índice da classe usando o mapa
            String classe = valores[4];
            int indiceClasse = mapaClasses.get(classe);
            saida[indiceClasse] = 1.0;

            entradas.add(entrada);
            saidas.add(saida);
        }
        br.close();
    }

    // Método para obter o índice do maior valor em um array
    private static int argMax(double[] array) {
        int index = 0;
        double max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                index = i;
            }
        }
        return index;
    }
}
