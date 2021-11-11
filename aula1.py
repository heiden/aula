# Desafio de casa: ver o que tem no valor de W (os pesos) e plotar a reta

import numpy as np

def funcao_ativacao(valor):
  if valor < 0:
    return -1
  else:
    return 1

peso = np.array([113, 122, 102, 98, 115, 120, 101, 107])
acidez = np.array([6.8, 4.7, 5.2, 5.6, 2.9, 4.2, 3.2, 5.0])

X = np.vstack((peso, acidez))
Y = np.array([-1, 1, -1, -1, 1, 1, 1, -1]) # maca -1, laranja +1

epocas = 100
taxa_de_aprendizado = 0.1
bias = 1

W = np.zeros([1, 3]) # dois atributos + o bias

erros = np.zeros(len(peso))

for i in range(epocas):
  for j in range(len(peso)):
    Xb = np.hstack((bias, X[:, j]))

    v = np.dot(W, Xb)

    y = funcao_ativacao(v)

    erros[j] = Y[j] - y

    W = W + taxa_de_aprendizado * erros[j] * Xb
  print(sum(erros))

# print(W)

maca = np.array([1, 122, 6.3])
laranja = np.array([1, 107, 3.8])

vm = np.dot(W, maca)
print('y para maca', funcao_ativacao(vm))

vl = np.dot(W, laranja)
print('y para laranja', funcao_ativacao(vl))
