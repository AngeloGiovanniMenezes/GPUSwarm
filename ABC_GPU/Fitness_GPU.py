import math
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

def fitness(numero_elementos, tamanho_grid, posicoes_X, posicoes_Y, vetor_fitness):

      #calculo do fitness se refere a funcao a ser analisada, nesse caso esta sendo utilizado a funcao de rastrigin
      
      posicao_X_particulas_GPU = cuda.mem_alloc(posicoes_X.nbytes)
      posicao_Y_particulas_GPU = cuda.mem_alloc(posicoes_Y.nbytes)
      cuda.memcpy_htod(posicao_X_particulas_GPU, posicoes_X)
      cuda.memcpy_htod(posicao_Y_particulas_GPU, posicoes_Y)      
      resultado_fitness_GPU = cuda.mem_alloc(numero_elementos * 32)
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
      
      fitness = kernel.get_function("fitness")
      fitness(posicao_X_particulas_GPU, posicao_Y_particulas_GPU, resultado_fitness_GPU, np.int32(1024), block = (1024, 1, 1), grid = (tamanho_grid + 1, 1, 1))

      cuda.memcpy_dtoh(vetor_fitness, resultado_fitness_GPU)
