import random
import math
import numpy as np
#import Funcoes_CPU
import time

coeficiente_inercia = 0.5
coeficiente_aceleracao1 = 0.8
coeficiente_aceleracao2 = 0.9

numero_iteracoes = 100
num_execucoes = 10

dimensoes = 2
valor_maximo_x = 10000
valor_maximo_y = 10000

numero_particulas = 20480

class Particula():

    def __init__(self, numero_particulas):

        self.numero_particulas = numero_particulas
        #inicia a particula em uma posicao aleatoria, onde o Z da funcao eh o fitness.
        #posicao sera representado por um float32, devido a velocidade de calculo desse tipo em GPU.
        self.posicao_X = np.float32([(-1) ** (bool(random.getrandbits(1))) * random.random() * valor_maximo_x])
        self.posicao_Y = np.float32([(-1)**(bool(random.getrandbits(1))) * random.random() * valor_maximo_y])
        self.posicao = np.column_stack((self.posicao_X, self.posicao_Y))
        
        self.pbest_posicao = np.column_stack((self.posicao_X, self.posicao_Y))
        self.pbest_valor = float('inf')
        self.velocidade = np.array([0,0])

    def printa_particula(self):

        print("Posicao: ", self.posicao, "Pbest: ", self.pbest_posicao)

    def converge(self):

        self.posicao = self.posicao + self.velocidade

class Espaco():

    def __init__(self, objetivo, erro_minimo, numero_particulas, funcao):

        self.objetivo = objetivo
        self.erro = erro_minimo
        self.numero_particulas = numero_particulas
        self.particulas = []
        self.gbest_valor = float('inf')
        self.gbest_posicao = np.array([random.random() * valor_maximo_x, random.random() * valor_maximo_y])
        self.funcao = funcao

    def printa_particulas(self):

        for particula in self.particulas:

            particula.printa_particula()

    def fitness(self, particula):
      
      #return 0.5 + ((np.sin((particula.posicao_X ** 2) - (particula.posicao_Y ** 2)) ** 2) - 0.5) / ((1 + 0.001) * (particula.posicao_X ** 2 + particula.posicao_Y ** 2)) ** 2
      return (particula.posicao_X ** 2 - 10 * np.cos(2 * math.pi * particula.posicao_X)) + (particula.posicao_Y ** 2 - 10 * np.cos(2 * math.pi * particula.posicao_Y)) + 20
      
    def set_pbest(self):

        for particula in self.particulas:

            fitness = self.fitness(particula)
            if(particula.pbest_valor > fitness):

                particula.pbest_valor = fitness
                particula.pbest_posicao = particula.posicao

    def set_gbest(self):

        for particula in self.particulas:

            melhor_fitness = self.fitness(particula)
            if(self.gbest_valor > melhor_fitness):

                self.gbest_valor =  melhor_fitness
                self.gbest_posicao = particula.posicao

    def converge_particulas(self):

        global coeficiente_inercia

        for particula in self.particulas:

            nova_velocidade = (coeficiente_inercia*particula.velocidade) + (coeficiente_aceleracao1*random.random()) * (particula.pbest_posicao - particula.posicao) + (coeficiente_aceleracao2*random.random()) * (self.gbest_posicao-particula.posicao)
            particula.velocidade = nova_velocidade
            particula.converge()

def loop(arquivo, iteracao_atual):

    tempo_inicio = time.time()

    espaco_solucao = Espaco(0.0, 0, numero_particulas, "rastrigin")
    vetor_particulas = [Particula(espaco_solucao.numero_particulas) for _ in range(espaco_solucao.numero_particulas)]
    espaco_solucao.particulas = vetor_particulas
    
    iteracoes = 0

    while(iteracoes < numero_iteracoes):

        espaco_solucao.set_pbest()
        espaco_solucao.set_gbest()
        espaco_solucao.converge_particulas()
        iteracoes += 1

    tempo_fim = time.time()

    arquivo.write("PSO/CPU Iteracao: " + str(iteracao_atual + 1) + "\nTempo de execucao: " + str(tempo_fim - tempo_inicio) + "\nMelhor fitness: " + str(espaco_solucao.gbest_valor))

    espaco_solucao = None
    vetor_particulas = None

def main():

	arquivo_tempos = open ("tempos", "w")

	for i in range(0, num_execucoes):

		loop(arquivo_tempos, i)
		
main()
