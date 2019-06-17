import random
import numpy as np
import pprint as pp
import time
import math
import Fitness_GPU
from operator import attrgetter
import Funcoes_CPU

numero_abelhas = 5120
numero_iteracoes = 100
numero_tentativas = 50
intervalo_superior = [10000, 10000]
intervalo_inferior = [-10000, -10000]
porcentagem_abelhas_empregadas = 0.5

numero_execucoes = 1

class Colmeia():

    fontes_alimento = []

    def __init__(self, numero_abelhas, numero_iteracoes, numero_tentativas, intervalo_inferior, intervalo_superior, porcentagem_abelhas_empregadas, funcao):

        self.numero_abelhas = numero_abelhas
        self.numero_iteracoes = numero_iteracoes
        self.numero_tentativas = numero_tentativas
        self.porcentagem_abelhas_empregadas = porcentagem_abelhas_empregadas

        self.valor_minimo_eixo = intervalo_inferior
        self.valor_maximo_eixo = intervalo_superior

        self.abelhas_empregadas = int(round(numero_abelhas * porcentagem_abelhas_empregadas))
        self.abelhas_olheiras = numero_abelhas - self.abelhas_empregadas
        self.valor_minimo_eixo = np.array(intervalo_inferior)
        self.valor_maximo_eixo = np.array(intervalo_superior)

        self.funcao = funcao

        self.posicoes_X = np.zeros(self.numero_abelhas)
        self.posicoes_X = self.posicoes_X.astype(np.float32)

        self.posicoes_Y = np.zeros(self.numero_abelhas)
        self.posicoes_Y = self.posicoes_Y.astype(np.float32)

        self.vetor_fitness = np.ndarray(self.numero_abelhas)
        self.vetor_fitness = self.vetor_fitness.astype(np.float32)

    def otimiza(self):
        
        self.inicializa()
        pp.pprint(self.fontes_alimento)

        for numero_iteracoes in range(1, numero_iteracoes + 1):
            
            self.fase_abelhas_empregadas()
            self.fase_abelhas_olheiras()
            self.fase_abelhas_campeiras()

        pp.pprint(self.fontes_alimento)

        melhor_fs = self.melhor_fonte()

        return melhor_fs.solucao

    def inicializa(self):

        self.fontes_alimento = [self.cria_fonte_alimento_inicial(i) for i in range(self.abelhas_empregadas)]
        tamanho_grid = int(math.ceil(self.numero_abelhas / 1024) + 1)        
        Fitness_GPU.fitness(self.numero_abelhas, tamanho_grid, self.posicoes_X, self.posicoes_Y, self.vetor_fitness)
        
        for i in range(0, len(self.fontes_alimento)):
            if self.vetor_fitness[i] >= 0:
                fitness = 1 / (1 + self.vetor_fitness[i])
            else:
                fitness = abs(self.vetor_fitness[i])
            self.fontes_alimento[i].set_fitness(fitness)

    def fase_abelhas_empregadas(self):

        for i in range(self.abelhas_empregadas):
            fontes_alimento = self.fontes_alimento[i]
            nova_solucao = self.gera_solucao(i)
            melhor_solucao = self.melhor_solucao(fontes_alimento.solucao, nova_solucao)

            self.set_solucao(fontes_alimento, melhor_solucao)

    def fase_abelhas_olheiras(self):

        for i in range(self.abelhas_olheiras):

            probabilidades = [self.probabilidade(fs) for fs in self.fontes_alimento]
            indice_selecionado = self.seleciona(range(len(self.fontes_alimento)), probabilidades)
            fonte_selecionada = self.fontes_alimento[indice_selecionado]
            nova_solucao = self.gera_solucao(indice_selecionado)
            melhor_solucao = self.melhor_solucao(fonte_selecionada.solucao, nova_solucao)

            self.set_solucao(fonte_selecionada, melhor_solucao)

    def fase_abelhas_campeiras(self):

        for i in range(self.abelhas_empregadas):
            fontes_alimento = self.fontes_alimento[i]

            if fontes_alimento.tentativas > self.numero_tentativas:
                fontes_alimento = self.cria_fonte_alimento()


    def gera_solucao(self, indice_solucao_atual):

        solucao = self.fontes_alimento[indice_solucao_atual].solucao
        k_source_index = self.solucao_aleatoria([indice_solucao_atual])
        k_solucao = self.fontes_alimento[k_source_index].solucao
        d = random.randint(0, len(self.valor_minimo_eixo) - 1)
        r = random.uniform(-1, 1)

        nova_solucao = np.copy(solucao)
        nova_solucao[d] = solucao[d] + r * (solucao[d] - k_solucao[d])

        return np.around(nova_solucao, decimals=4)

    def solucao_aleatoria(self, indice_ignorado):

        indices_disponiveis = set(range(self.abelhas_empregadas))
        ignorados = set(indice_ignorado)
        diferenca = indices_disponiveis - ignorados
        selecionado = random.choice(list(diferenca))

        return selecionado

    def melhor_solucao(self, current_solucao, nova_solucao):

        if self.fitness(nova_solucao) > self.fitness(current_solucao):
            return nova_solucao
        else:
            return current_solucao

    def probabilidade(self, solucao_fitness):

        soma_fitness = sum([fs.fitness for fs in self.fontes_alimento])
        probabilidade = solucao_fitness.fitness / soma_fitness

        return probabilidade

    def fitness(self, solucao):

        result = Funcoes_CPU.fitness_CPU(solucao[0], solucao[1])

        if result >= 0:
            fitness = 1 / (1 + result)
        else:
            fitness = abs(result)

        return fitness

    def seleciona(self, solucao, weights):

        return random.choices(solucao, weights)[0]

    def set_solucao(self, fontes_alimento, nova_solucao):

        if np.array_equal(nova_solucao, fontes_alimento.solucao):
            fontes_alimento.tentativas += 1
        else:
            fontes_alimento.solucao = nova_solucao
            fontes_alimento.tentativas = 0

    def melhor_fonte(self):

        melhor = max(self.fontes_alimento, key=attrgetter('fitness'))

        return melhor

    def cria_fonte_alimento(self):

        solucao = self.solucao_possivel(self.valor_minimo_eixo, self.valor_maximo_eixo)
        fitness = self.fitness(solucao)

        return FonteAlimento(solucao, fitness)

    def cria_fonte_alimento_inicial(self, i):

        solucao = self.solucao_possivel(self.valor_minimo_eixo, self.valor_maximo_eixo)
        self.posicoes_X[i] = solucao[0]
        self.posicoes_Y[i] = solucao[1]
        fitness = self.fitness(solucao)

        return FonteAlimento(solucao, fitness)
    def solucao_possivel(self, lb, ub):

        r = random.random()
        solucao = lb + (ub - lb) * r
        return np.around(solucao, decimals=4)

    def run(self):

        iteracao_atual = 0
        self.inicializa()
        while (iteracao_atual < self.numero_iteracoes):
            self.fase_abelhas_empregadas()
            self.fase_abelhas_campeiras()
            self.fase_abelhas_olheiras()
            print("Iteracao: ", iteracao_atual, "Melhor fitness: ", self.melhor_fonte().fitness, "Posicao: ", self.melhor_fonte().solucao)
            iteracao_atual += 1

class FonteAlimento(object):

    def __init__(self, solucao_inicial, fitness_inicial):

        super(FonteAlimento, self).__init__()
        self.id = id
        self.solucao = solucao_inicial
        self.fitness = fitness_inicial
        self.tentativas = 0

    def set_fitness(self, novo_fitness):
        self.fitness = novo_fitness

    def get_fitness(self):
        return self.fitness

def loop(arquivo, iteracao_atual):

    tempo_inicio = time.time()

    ABC = Colmeia(numero_abelhas, numero_iteracoes, numero_tentativas, intervalo_inferior, intervalo_superior, porcentagem_abelhas_empregadas, 'rastrigin')
    ABC.run()

    tempo_fim = time.time()

    arquivo.write("\tABC/GPU Iteracao: " + str(iteracao_atual + 1) + "\nTempo de execucao: " + str(tempo_fim - tempo_inicio) + "\nMelhor fitness: " + str(ABC.melhor_fonte().fitness))

    ABC = None

def main():

    arquivo = open("tempos_ABC_CPU", "w")

    for i in range(0, numero_execucoes):
        loop(arquivo, i)

main()
