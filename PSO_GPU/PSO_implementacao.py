import random
import math
import numpy as np
import Fitness_GPU
import Funcoes_CPU
import time

coeficiente_inercia = 0.5
coeficiente_aceleracao1 = 0.8
coeficiente_aceleracao2 = 0.9

numero_iteracoes = 100
num_execucoes = 10
numero_particulas = 20480

dimensoes = 2
valor_maximo_x = 10000
valor_maximo_y = 10000

class Particula():

      def __init__(self, numero_particulas, funcao):

            self.funcao = funcao
            self.numero_particulas = numero_particulas
            #inicia a particula em uma posicao aleatoria, onde o Z da funcao eh o fitness.
            #posicao sera representado por um float32, devido a velocidade de calculo desse tipo em GPU.
            self.posicao_X = np.float32([(-1) ** (bool(random.getrandbits(1))) * random.random() * valor_maximo_x])
            self.posicao_X = self.posicao_X.astype(np.float32)
            self.posicao_Y = np.float32([(-1)**(bool(random.getrandbits(1))) * random.random() * valor_maximo_y])
            self.posicao_Y = self.posicao_Y.astype(np.float32)
            self.posicao = np.column_stack((self.posicao_X, self.posicao_Y))
        
            self.pbest_posicao = np.column_stack((self.posicao_X, self.posicao_Y))
            self.pbest_valor = np.float32('inf')
            self.velocidade = np.array([0,0])

      def printa_particula(self):

            print("Posicao: ", self.posicao, "Pbest: ", self.pbest_posicao)

      def converge(self):

            self.posicao = self.posicao + self.velocidade

class Espaco():

    def __init__(self, objetivo, erro_minimo, numero_particulas):
        
        self.tamanho_grid = int(math.ceil(numero_particulas / 1024) + 1)
        self.fitness = np.empty(numero_particulas)
        self.objetivo = objetivo
        self.erro = erro_minimo
        self.numero_particulas = numero_particulas
        self.particulas = []
        self.gbest_valor = float('inf')
        self.gbest_posicao = np.array([random.random() * valor_maximo_x, random.random() * valor_maximo_y])

    def printa_particulas(self):

        for particula in self.particulas:
            particula.printa_particula()

    def fitness_todas_particulas(particulas):

        Fitness_GPU.fitness_todas_particulas_GPU(particulas)
    
    def fitness(self, particula):

        X = float(particula.posicao_X)
        Y = float(particula.posicao_Y)

        result = (X ** 2 - 10 * math.cos(2 * math.pi * X)) + (Y ** 2 - 10 * math.cos(2 * math.pi * Y)) + 20
        return result

    def set_pbest(self, posicoes_X, posicoes_Y, particulas):

        Fitness_GPU.fitness_todas_particulas_GPU(self, posicoes_X, posicoes_Y, particulas)

    def set_gbest(self):

        for i in range(0, self.numero_particulas):
            melhor_fitness = self.fitness[i]
            if(self.gbest_valor > melhor_fitness):

                self.gbest_valor =  melhor_fitness
                self.gbest_posicao = self.particulas[i].posicao

    def converge_particulas(self):

        global coeficiente_inercia

        for particula in self.particulas:

            nova_velocidade = (coeficiente_inercia*particula.velocidade) + (coeficiente_aceleracao1*random.random()) * (particula.pbest_posicao - particula.posicao) + (coeficiente_aceleracao2*random.random()) * (self.gbest_posicao-particula.posicao)
            particula.velocidade = nova_velocidade
            particula.converge()

def get_posicoes_X(posicoes_X, vetor_particulas):

    for i in range(0, vetor_particulas[0].numero_particulas):
        posicoes_X[i] = vetor_particulas[i].posicao_X

def get_posicoes_Y(posicoes_Y, vetor_particulas):

    for i in range(0, vetor_particulas[0].numero_particulas):
        posicoes_Y[i] = vetor_particulas[i].posicao_Y

def loop(arquivo, iteracao_atual):

      #erro_minimo = float(input("Digite o valor minimo do percentual de erro: "))
      tempo_inicio = time.time()

      espaco_solucao = Espaco(0.0, 0, numero_particulas)
      vetor_particulas = [Particula(espaco_solucao.numero_particulas, "rastrigin") for _ in range(espaco_solucao.numero_particulas)]
      
      tempo_aux = time.time()
      tempo_meio = tempo_aux - tempo_inicio
      
      #CUSTO ADICIONAL DE ITERAR SOBRE O VETOR DE POSICOES
      posicoes_X = np.zeros(vetor_particulas[0].numero_particulas)
      posicoes_Y = np.zeros(vetor_particulas[0].numero_particulas)
      posicoes_X = posicoes_X.astype(np.float32)
      posicoes_Y = posicoes_Y.astype(np.float32)
      get_posicoes_X(posicoes_X, vetor_particulas)
      get_posicoes_Y(posicoes_Y, vetor_particulas)
      espaco_solucao.particulas = vetor_particulas
      
      tempo_aux = time.time()

      iteracoes = 0

      while(iteracoes < numero_iteracoes):

            espaco_solucao.set_pbest(posicoes_X, posicoes_Y, vetor_particulas)
            espaco_solucao.set_gbest()
            espaco_solucao.converge_particulas()
            iteracoes += 1

      tempo_fim = time.time()

      arquivo.write("\tPSO/GPU Iteracao: " + str(iteracao_atual + 1) + "\nTempo de execucao: " + str(tempo_meio + (tempo_fim - tempo_aux)) + "\nMelhor fitness: " + str(espaco_solucao.gbest_valor))

      espaco_solucao = None
      vetor_particulas = None
      posicoes_X = None
      posicoes_Y = None

def main():

	arquivo_tempos = open ("tempos", "w")

	for i in range(0, num_execucoes):

		loop(arquivo_tempos, i)
		
main()
