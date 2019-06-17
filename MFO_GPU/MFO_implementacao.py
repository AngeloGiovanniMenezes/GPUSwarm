import random
import numpy as np
import math
import time
import Funcoes_CPU
import Fitness_GPU

numero_iteracoes = 100
numero_execucoes = 10
numero_mariposas = 20480
dimensao = 2

class MFO():

    def __init__(self, funcao, valor_minimo_eixo, valor_maximo_eixo):

        self.funcao = funcao
        self.valor_minimo_eixo = valor_minimo_eixo
        self.valor_maximo_eixo = valor_maximo_eixo
        self.posicao_mariposa = np.zeros((numero_mariposas, dimensao))

        if not isinstance(self.valor_minimo_eixo, list):
            self.valor_minimo_eixo = [self.valor_minimo_eixo] * dimensao
        if not isinstance(self.valor_maximo_eixo, list):
            self.valor_maximo_eixo = [valor_maximo_eixo] * dimensao

        for i in range(dimensao):
            self.posicao_mariposa[:, i] = np.random.uniform(0, 1, numero_mariposas) * (self.valor_maximo_eixo[i] - self.valor_minimo_eixo[1]) + self.valor_minimo_eixo[i]
        
        self.curva_convergencia = np.zeros(numero_iteracoes)
        self.posicoes_X = np.ones(numero_mariposas)
        self.posicoes_X = self.posicoes_X.astype(np.float32)

        self.posicoes_Y = np.ones(numero_mariposas)
        self.posicoes_Y = self.posicoes_Y.astype(np.float32)

        self.fitness_mariposa = np.ndarray(numero_mariposas)
        self.fitness_mariposa = self.fitness_mariposa.astype(np.float32)

        self.populacao_ordenada = np.copy(self.posicao_mariposa)
        self.fitness_ordenada = np.zeros(numero_mariposas)

        self.melhores_luzes = np.copy(self.posicao_mariposa)
        self.fitness_melhor_luz = np.zeros(numero_mariposas)

        self.dobro_populacao = np.zeros((2 * numero_mariposas, dimensao))
        self.dobro_fitness = np.zeros(2 * numero_mariposas)

        self.dobro_populacao_ordenada = np.zeros((2 * numero_mariposas, dimensao))
        self.dobro_fitness_ordenada = np.zeros(2 * numero_mariposas)

        self.populacao_anterior = np.zeros((numero_mariposas, dimensao))
        self.fitness_anterior = np.zeros(numero_mariposas)

    def run(self):

        iteracao = 1
        while (iteracao < numero_iteracoes):

            numero_luzes = round((numero_mariposas - iteracao * ((numero_mariposas - 1)/numero_iteracoes)))

            for i in range(0, numero_mariposas):
                for j in range(dimensao):
                    self.posicao_mariposa[i, j] = np.clip(self.posicao_mariposa[i, j], self.valor_minimo_eixo[j], self.valor_maximo_eixo[j])

                self.posicoes_X[i] = np.float32(self.posicao_mariposa[i, 0])
                self.posicoes_Y[i] = np.float32(self.posicao_mariposa[i, 1])
            
            tamanho_grid = int(math.ceil(numero_mariposas / 1024) + 1)
            Fitness_GPU.fitness(numero_mariposas, tamanho_grid, self.posicoes_X, self.posicoes_Y, self.fitness_mariposa)
            if iteracao == 1:
                self.fitness_ordenado = np.sort(self.fitness_mariposa)
                I = np.argsort(self.fitness_mariposa)

                self.populacao_ordenada = self.posicao_mariposa[I,:]

                self.melhores_luzes = self.populacao_ordenada
                self.melhores_luzes_fitness = self.fitness_ordenado
            else:
                self.dobro_populacao = np.concatenate((self.populacao_anterior, self.melhores_luzes), axis = 0)
                self.dobro_fitness = np.concatenate((self.fitness_anterior, self.melhores_luzes_fitness), axis = 0)
                self.dobro_fitness_ordenado = np.sort(self.dobro_fitness)
                I2 = np.argsort(self.dobro_fitness)
            
                for novo_indice in range(0, 2 * numero_mariposas):
                    self.dobro_populacao_ordenada[novo_indice, :] = np.array(self.dobro_populacao[I2[novo_indice], :])
                
                self.fitness_ordenado = self.dobro_fitness_ordenado[0 : numero_mariposas]
                self.populacao_ordenada = self.dobro_populacao_ordenada[0 : numero_mariposas,:]
            
                self.melhores_luzes = self.populacao_ordenada
                self.melhor_luz_fitness = self.fitness_ordenada

            self.melhor_luz_valor = self.fitness_ordenado[0]
            self.melhor_luz_posicao = self.populacao_ordenada[0, :]

            self.populacao_anterior = self.posicao_mariposa
            self.fitness_anterior = self.fitness_mariposa

            a = - 1 + iteracao * ((-1) / numero_iteracoes)

            for i in range(0, numero_mariposas):
                for j in range(0, dimensao):
                    if(i <= numero_luzes):
                        distancia_ate_chama = abs(self.populacao_ordenada[i, j] - self.posicao_mariposa[i,j])
                        b = 1
                        t = (a - 1) * random.random() + 1
                        self.posicao_mariposa[i,j] = distancia_ate_chama * math.exp(b * t) * math.cos(t * 2 * math.pi) + self.populacao_ordenada[i, j]

            self.curva_convergencia[iteracao] = self.melhor_luz_valor

            iteracao += 1

    def get_melhor_fitness(self):

        return self.melhor_luz_valor

def loop(arquivo, iteracao_atual):

    #erro_minimo = float(input("Digite o valor minimo do percentual de erro: "))
    tempo_inicio = time.time()
        
    mariposas = MFO('rastrigin', -10000, 10000)
    mariposas.run()
    tempo_fim = time.time()

    arquivo.write("\tMFO/CPU Iteracao: " + str(iteracao_atual + 1) + "\nTempo de execucao: " + str(tempo_fim - tempo_inicio) + "\nMelhor fitness: " + str(mariposas.get_melhor_fitness()))

    mariposas = None

def main():

    arquivo_tempos = open ("tempos", "w")

    for i in range(0, numero_execucoes):

        loop(arquivo_tempos, i)
        
main()
