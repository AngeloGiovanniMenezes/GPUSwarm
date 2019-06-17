import math
import numpy as np
import random

valor_maximo_x = 100000
valor_maximo_y = 100000

def gera_posicao_CPU(elemento):

	elemento.posicao =  np.array([(-1) ** (bool(random.getrandbits(1))) * random.random() * valor_maximo_x, (-1)**(bool(random.getrandbits(1))) * random.random() * valor_maximo_y])

def fitness_CPU(elemento):
        #calculo do fitness se refere a funcao a ser analisada, nesse caso esta sendo utilizado a funcao de rastrigin
		
		#RASTRIGIN        
		return (elemento.posicao[0] ** 2 - 10 * math.cos(2 * math.pi * elemento.posicao[0])) + (elemento.posicao[1] ** 2 - 10 * math.cos(2 * math.pi * elemento.posicao[1])) + 20

def rastrigin(d):
    sum_i = np.sum([x**2 - 10*np.cos(2 * np.pi * x) for x in d])
    return 10 * len(d) + sum_i

def rastriginFn(d):
	def fn(*args):
		sum_i = sum([args[i]**2 - 10*np.cos(2 * np.pi * args[i]) for i in range(len(args))])
		return 10 * d + sum_i
	return fn

"""
def sphere(d):
    return np.sum([x**2 for x in d])

def sphereFn(d):
    def fn(*args):
        return sum([args[i]**2 for i in range(len(args))])
    return fn

def ackley(d, *, a=20, b=0.2, c=2*np.pi):
    sum_part1 = np.sum([x**2 for x in d])
    part1 = -1.0 * a * np.exp(-1.0 * b * np.sqrt((1.0/len(d)) * sum_part1))

    sum_part2 = np.sum([np.cos(c * x) for x in d])
    part2 = -1.0 * np.exp((1.0 / len(d)) * sum_part2)

    return a + np.exp(1) + part1 + part2

def ackleyFn(d, *, a=20, b=0.2, c=2*np.pi):

	def fn(*args):

		sum_part1 = sum([args[i]**2 for i in range(len(args))])
		part1 = -1.0 * a * np.exp(-1.0 * b * np.sqrt((1.0/d) * sum_part1))
		sum_part2 = sum([np.cos(c * args[i]) for i in range(len(args))])
		part2 = -1.0 * np.exp((1.0 / d) * sum_part2)
		return a + np.exp(1) + part1 + part2
	return fn
"""
