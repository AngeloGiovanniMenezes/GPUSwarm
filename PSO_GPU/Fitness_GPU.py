import math
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

def fitness_todas_particulas_GPU(espaco, posicoes_X, posicoes_Y, particulas):

      #calculo do fitness se refere a funcao a ser analisada, nesse caso esta sendo utilizado a funcao de rastrigin
      
      transferencia_inicio = time.time()

      posicao_X_particulas_GPU = cuda.mem_alloc(posicoes_X.nbytes)
      posicao_Y_particulas_GPU = cuda.mem_alloc(posicoes_Y.nbytes)
      cuda.memcpy_htod(posicao_X_particulas_GPU, posicoes_X)
      cuda.memcpy_htod(posicao_Y_particulas_GPU, posicoes_Y)
      
      resultado_fitness_GPU = cuda.mem_alloc(espaco.numero_particulas * 32)
      
      transferencia_fim = time.time()
      
      result = transferencia_fim - transferencia_inicio
      #print("Tempo transferencia", result)

      kernel = SourceModule("""

            #include <math.h>

            __global__ void fitness(float *posicao_X_particulas_GPU, float *posicao_Y_particulas_GPU, float *resultado_fitness_GPU, int max_i)
            {
                  int idx = blockIdx.x *blockDim.x + threadIdx.x;
                  float pi = 3.14150265358979323846;
                  
                  if(idx < max_i)
                  {
                        resultado_fitness_GPU[idx] = pow(posicao_X_particulas_GPU[idx], 2) - (10 *  cos(2 * pi * posicao_X_particulas_GPU[idx])) + (pow(posicao_Y_particulas_GPU[idx], 2) - (10 * cos(2 * pi * posicao_Y_particulas_GPU[idx]))) + 20;
                  }
            }
      """)
      
      execucao_inicio = time.time()
      fitness = kernel.get_function("fitness")
      fitness(posicao_X_particulas_GPU, posicao_Y_particulas_GPU, resultado_fitness_GPU, np.int32(espaco.numero_particulas), block = (1024, 1, 1), grid = ((espaco.tamanho_grid + 1), 1, 1))
      execucao_fim = time.time()
      
      #print("Tempo execucao" , (execucao_fim - execucao_inicio))
      
      resultado_inicio = time.time()
      resultado_fitness = np.ndarray(espaco.numero_particulas)
      resultado_fitness = resultado_fitness.astype(np.float32)
      cuda.memcpy_dtoh(resultado_fitness, resultado_fitness_GPU)
      resultado_fim = time.time()
      #print("Resultado tempo: ", (resultado_fim - resultado_inicio))
      espaco.fitness = resultado_fitness
      
      atualiza_pbest(particulas, resultado_fitness)

def atualiza_pbest(particulas, resultado_fitness):

      #NAO-PARALELO
      for i in range(0, particulas[0].numero_particulas):
            if (particulas[i].pbest_valor > resultado_fitness[i]):
                  particulas[i].pbest_valor = resultado_fitness[i]
                  particulas[i].pbest_posicao = particulas[i].posicao
