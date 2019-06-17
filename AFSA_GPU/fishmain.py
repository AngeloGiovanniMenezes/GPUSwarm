import random
import math
import time
import numpy as np
from cfish import *
#from scipy.spatial import distance
import copy

#algo params
dim=2
population = 1024
GroupFish=[]
trytimes=3
#Fish params
Visual=0.2
step=0.3

#params
iteration=100

numero_execucoes = 1

def loop(arquivo, iteracao_atual):

	tempo_inicio = time.time()
	StoreBest=[]
	#init Fish 
	initialize(dim, population, GroupFish)

	B=getBestFish(GroupFish)
	StoreBest.append(copy.deepcopy(B))

	i = 0
	while i< iteration:
		j=0
		while j<population:
			k=0
			while k<trytimes:	
				temp_Position=makeTemp(GroupFish[j], Visual)
				if GroupFish[j].fitness>temp_Position.fitness:
					prey(GroupFish[j], temp_Position, B, dim, step, population, Visual, GroupFish, j)
					break
				k=k+1
			moveRandomly(GroupFish[j], Visual)
			j=j+1
			#leapFish(GroupFish)
		i=i+1
		B=getBestFish(GroupFish)
		StoreBest.append(copy.deepcopy(B))

	BE=getBestFish(StoreBest)

	tempo_fim = time.time()

	arquivo.write("\tAFSA/GPU Iteracao: " + str(iteracao_atual + 1) + "\nTempo de execucao: " + str(tempo_fim - tempo_inicio) + "\nMelhor fitness: " + str(BE.fitness))

def main():

    arquivo = open("tempos", "w")

    for i in range(0, numero_execucoes):
        loop(arquivo, i)

main()
