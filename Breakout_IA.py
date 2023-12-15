import gymnasium as gym
import sys
from random import randint
import numpy as np
import cv2  # Necesitarás OpenCV para trabajar con imágenes

# Nuestro espacio es continuo (O sea es un espacio con arreglos multidimensinales),
# por lo que debemos transformarlo a discreto.
def discretizar_estado(estado):
   matriz = estado[0] # Obtencion matriz.
   matriz_recortada = matriz[17:-9, 8:-8] # Se recorta la matriz
   matriz_pequeña = cv2.resize(matriz_recortada, (92, 72)) # Se reajusta la resolucion a la mitad.
   arreglo = np.where(matriz_pequeña != 0, 1, 0).flatten() # Se cambian los valores (0s quedan como 0 y diferentes quedan
                                                           # como 1) y se pasa a arreglo.
   np.set_printoptions(threshold=sys.maxsize) # Imprime el arreglo resultante
   print(arreglo)

   return arreglo

# Iniacializacion del entorno.
env = gym.make("ALE/Breakout-v5", render_mode = "human", obs_type = "grayscale")
env.reset()

# Inicializacion de la Q-Table.
num_estados = env.observation_space.shape[0]
num_acciones = env.action_space.n
Q_table = np.zeros((num_estados, num_acciones))

# Parametros del Q-Learning.
tasa_aprendizaje = 0.1
factor_descuento = 0.9
prob_accion_rand = 0.1
repeticiones = 2

# Bucle Q-Learning.
for rep in range(repeticiones):
   e = env.reset()
   discretizar_estado(e)
   #estado_discretizado = discretizar_estado(estado_continuo)

   final = False
   while not final:
      """
      if np.random.rand() < prob_accion_rand:
         accion = env.action_space.sample()  # Exploración aleatoria
      else:
         accion = np.argmax(Q_table[estado, :])  # Explotación de la Q-Table
      """
      accion = randint(0,3)

      # Realiza la acción en el entorno
      proximo_estado, recompensa, final, misterio, diccionario = env.step(accion)
      # Actualiza la Q-Table usando la fórmula de actualización Q
      # Q_table[estado, accion] = (1 - tasa_aprendizaje) * Q_table[estado, accion] + \
      #                          tasa_aprendizaje * (recompensa + factor_descuento * np.max(Q_table[proximo_estado, :]))

      #estado = proximo_estado

         # Renderiza el entorno (puedes comentar esto para mejorar la velocidad)

      env.render()

env.close()