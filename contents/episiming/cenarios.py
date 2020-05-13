#!/anaconda3/envs/py38/bin/python
# -*- coding: utf-8 -*-
"""
Módulo com cenários para simulações de modelos epidemiológicos.
"""

from collections import namedtuple

from functools import partial

import random

import numpy as np
from scipy.integrate import solve_ivp

import networkx as nx

from episiming import redes, individuais

def power_decay(a, b, x):
    return 1.0/(1.0 + (x/a)**b)

class Cenario:
    def __init__(self, num_pop, num_infectados_0, beta, gamma):
        self.nome = 'Base'
        self.define_parametros()
        self.cria_redes()
        self.inicializa_pop_estado()

    def define_parametros(self):
        self.num_pop = 1
        self.beta = 1
        self.gamma = 1
        self.attr_pos = {0: [0.0, 0.0]}
        self.pop_posicoes = np.array([[0.0, 0.0]])
        self.f_kernel = partial(power_decay, 1.0, 1.0)

    def cria_redes(self):
        """
        Gera uma rede regular completa.

        Deve ser sobre-escrito para cenários diferentes.
        """
        self.redes = []
        self.redes_tx_transmissao = []
        self.pop_fator_tx_transmissao_c = np.array([0])
        

    def inicializa_pop_estado(self, num_infectados_0):
        """
        Distribui aleatoriamente um certo número de infectados.
        """
        self.pop_estado_0 = np.array([1])
        self.attr_estado_0 = {0: {'estado': 1}}


    def exibe_redes(self, info=True, node_size=100, hist = True):
        for G in self.redes:
            redes.analise_rede(G, info=info, node_size=node_size,
                               pos=self.attr_pos, hist=hist)

    def evolui(self, dados_temporais, num_sim, show=''):
        X = individuais.evolucao_vetorial(
                self.pop_estado_0, 
                self.pop_posicoes, 
                self.redes, 
                self.redes_tx_transmissao,
                self.pop_fator_tx_transmissao_c,
                self.gamma,
                self.f_kernel,
                dados_temporais,
                num_sim,
                show
            )
        return X

    def evolui_matricial(self, dados_temporais, num_sim, show=''):
        tempos = np.linspace(
            dados_temporais[0],
            dados_temporais[1]*dados_temporais[2],
            dados_temporais[2] + 1
        )

        G_preps = list()
        for r in range(len(self.redes)):
            G_preps.append(self.redes[r].copy())
            attr_transmissao_edge = {
                (i, j): {'taxa de transmissao': self.redes_tx_transmissao[r][i]}
                       for (i,j) in self.redes[r].edges()
            }
            nx.set_edge_attributes(G_preps[-1], attr_transmissao_edge)

        G_c = nx.random_geometric_graph(self.num_pop, 0, pos=self.attr_pos)
        distancia = lambda x, y: sum(abs(a - b)**2 for a, b in zip(x, y))**0.5
        attr_kernel_dist = [(i, j, 
                     {'weight': self.f_kernel(
                         distancia(self.attr_pos[i], self.attr_pos[j]))}) 
                    for i in range(self.num_pop)
                    for j in range(self.num_pop) if j != i]

        G_c.add_edges_from(attr_kernel_dist)

        attr_transmissao = {(i, j): 
                            {'taxa de transmissao': self.beta_c * G_c.edges[i,j]['weight']/ G_c.degree(i, weight='weight')}
                            for (i,j) in G_c.edges()
                        }

        nx.set_edge_attributes(G_c, attr_transmissao)

        G_preps.append(G_c)
        G = nx.compose_all(G_preps)

        for (u, v, w) in G_c.edges.data('taxa de transmissao', default=0):
            G.edges[u,v]['taxa de transmissao'] = \
                G_c.edges[u,v]['taxa de transmissao']
            for G_aux in G_preps:
                if (u, v) in G_aux.edges:
                    G.edges[u, v]['taxa de transmissao'] += \
                        G_aux.edges[u,v]['taxa de transmissao']

        X = individuais.evolucao_matricial(
                self.pop_estado_0,
                G,
                self.gamma, 
                tempos,
                num_sim,
                show)
        return X

class RedeCompleta(Cenario):
    def __init__(self, num_pop, num_infectados_0, beta, gamma):
        self.nome = 'rede completa'
        self.define_parametros(num_pop, beta, gamma)
        self.inicializa_pop_estado(num_infectados_0)
        self.cria_redes()

    def define_parametros(self, num_pop, beta, gamma):
        self.num_pop = num_pop
        self.beta = beta

        self.beta_r = 0.0
        self.beta_s = beta
        self.beta_c = 0.0

        self.gamma = gamma

        self.pop_rho = np.ones(num_pop)

        self.f_kernel = partial(power_decay, 1.0, 1.5)

        self.attr_pos = dict()
        k = 0
        for i in range(self.num_pop):
            self.attr_pos.update(
                {k: [np.random.rand(), np.random.rand()]})
            k += 1
        self.pop_posicoes = np.array(list(self.attr_pos.values()))

    def inicializa_pop_estado(self, num_infectados_0):
        """
        Distribui aleatoriamente um certo número de infectados.
        """
        self.num_infectados_0 = num_infectados_0
        np.random.seed(seed = 342)
        #self.pop_estado_0 = np.ones(num_pop, dtype=np.uint8)
        self.pop_estado_0 = np.ones(self.num_pop)
        infectados_0 = np.random.choice(self.num_pop,
                                        self.num_infectados_0, 
                                        replace=False)
        #self.pop_estado_0[infectados_0] = \
        #    2*np.ones(num_infectados_0, dtype=np.uint8)
        self.pop_estado_0[infectados_0] = 2*np.ones(self.num_infectados_0)
        self.attr_estado_0 = dict([(i, {'estado': int(self.pop_estado_0[i])}) 
                                   for i in range(self.num_pop)])

    def cria_redes(self):
        """
        Gera uma rede completa.
        """
        G_reg = nx.random_regular_graph(d=self.num_pop-1,
                                        n=self.num_pop)

        nx.set_node_attributes(
            G_reg, 
            dict([(i, {'estado': int(self.pop_estado_0[i])})
                  for i in range(self.num_pop)])
            )

        nx.set_node_attributes(
            G_reg,
            dict([(i, {'rho': self.pop_rho[i]}) 
                  for i in range(self.num_pop)])
            )

        tx_transmissao_reg = np.array(
                [self.beta / (1+G_reg.degree(i)) 
                 for i in G_reg.nodes]
            )

        attr_transmissao = \
            dict([(i, {'taxa de transmissao': tx_transmissao_reg[i]}) 
                  for i in G_reg.nodes])

        nx.set_node_attributes(
            G_reg,
            dict([(i, {'taxa de transmissao': tx_transmissao_reg[i]}) 
                  for i in G_reg.nodes])
            )

        self.redes = [G_reg]
        self.redes_tx_transmissao = [tx_transmissao_reg]

        aux = np.array(
            [
                np.sum(
                    self.f_kernel(
                        np.linalg.norm(
                            self.pop_posicoes - self.pop_posicoes[i], axis=1)
                        )
                    ) 
                    for i in range(self.num_pop)
            ]
        )
        self.pop_fator_tx_transmissao_c = self.beta_c / aux
        
class Pop350(Cenario):

    def __init__(self):
        self.nome = 'Pop 350'
        self.define_parametros()
        self.inicializa_pop_estado()
        self.cria_redes()

    def define_parametros(self):

        # posições
        self.populacao_por_area = np.array(
            [
                [16, 11, 0, 0,  0,  6,  4,  8,  8,  6],
                [10, 12, 12, 6, 8, 9,  8,  6,  7,  5],
                [0, 10, 14, 10, 12,  8,  0,  0,  6,  8],
                [0, 12, 10, 14, 11,  9,  0,  0,  5,  7],
                [9, 11, 0, 12, 10,  7,  8,  7,  8, 0]
            ])

        self.num_pop = self.populacao_por_area.sum()

        self.pop_estado_0 = np.ones(self.num_pop)

        np.random.seed(seed = 127)
        self.attr_pos = dict()
        k = 0
        N, M = self.populacao_por_area.shape
        for m in range(M):
            for n in range(N):
                for i in range(self.populacao_por_area[n,m]):
                    self.attr_pos.update(
                        {k: [m + np.random.rand(), N - n + np.random.rand()]})
                    k += 1
        self.pop_posicoes = np.array(list(self.attr_pos.values()))

        # idades
        self.censo_fracoes = [0.15079769082144653, # 0 a 9 anos
                              0.17906470565542282, # 10 a 19 anos
                              0.18007108135150324, # 20 a 29 anos
                              0.15534569934620965, # 30 a 39 anos
                              0.13023309451263393, # 40 a 49 anos
                              0.09654553673621215, # 50 a 59 anos
                              0.059499784853198616, # 60 a 69 anos
                              0.033053176013799715, # 70 a 79 anos
                              0.015389230709573343] # 80 ou mais

        self.pop_idades = \
            np.random.choice(9, self.num_pop, p=self.censo_fracoes)

        self.raio_residencial = 0.6

        self.alpha_r = 0.8

        zeta_idade = lambda x: power_decay(50.0, 2.0, abs(x-35))

        self.distribuicao_social = [
            10, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
            ]
        self.rho_forma = 0.2 # shape factor of gamma distribution
        self.rho_escala = 5 # scale (mean value = scale * shape)
        self.pop_rho = np.random.gamma(self.rho_forma, self.rho_escala,
                                       self.num_pop) 

        self.a_kernel = 1.0
        self.b_kernel = 1.5
        self.f_kernel = partial(power_decay, self.a_kernel, self.b_kernel)

        self.num_infectados_0 = 8

        self.beta_r = 0.16
        self.beta_s = 0.24
        self.beta_c = 0.04
        self.gamma = 0.1        

    def inicializa_pop_estado(self):
        np.random.seed(seed = 342)
        #self.pop_estado_0 = np.ones(num_pop, dtype=np.uint8)
        self.pop_estado_0 = np.ones(self.num_pop)
        infectados_0 = np.random.choice(self.num_pop,
                                        self.num_infectados_0, 
                                        replace=False)
        #self.pop_estado_0[infectados_0] = \
        #    2*np.ones(num_infectados_0, dtype=np.uint8)
        self.pop_estado_0[infectados_0] = 2*np.ones(self.num_infectados_0)
        self.attr_estado_0 = dict([(i, {'estado': int(self.pop_estado_0[i])}) 
                                   for i in range(self.num_pop)])

    def cria_rede_residencial(self):

        self.G_r = nx.random_geometric_graph(
            self.num_pop, self.raio_residencial,
            pos=self.attr_pos, seed=1327)

        nx.set_node_attributes(
            self.G_r, 
            dict([(i, {'estado': int(self.pop_estado_0[i])})
                  for i in range(self.num_pop)])
            )

        nx.set_node_attributes(
            self.G_r,
            dict([(i, {'faixa etária': self.pop_idades[i]})
                  for i in range(self.num_pop)])
            )

        nx.set_node_attributes(
            self.G_r,
            dict([(i, {'rho': self.pop_rho[i]}) 
                  for i in range(self.num_pop)])
            )

        self.pop_tx_transmissao_r = np.array(
            [self.beta_r / (1+self.G_r.degree(i))**self.alpha_r 
            for i in self.G_r.nodes]
            )
        attr_transmissao_r = \
            dict([(i, {'taxa de transmissao': self.pop_tx_transmissao_r[i]}) 
                  for i in self.G_r.nodes])

        nx.set_node_attributes(
            self.G_r,
            dict([(i, {'taxa de transmissao': self.pop_tx_transmissao_r[i]}) 
                  for i in self.G_r.nodes])
            )

        nx.set_edge_attributes(self.G_r, 1, 'weight')

    def cria_rede_social(self):
        self.G_s = nx.random_geometric_graph(self.num_pop, 0,
                                             pos=self.attr_pos)
        nx.set_node_attributes(self.G_s, self.attr_estado_0)

        random.seed(721)
        pop_index = list(range(self.num_pop))
        membros = dict()

        for j in range(len(self.distribuicao_social)):
            individuos_aleatorios = \
                random.sample(pop_index, self.distribuicao_social[j])
            for i in individuos_aleatorios:
                pop_index.remove(i)
            membros.update({j: individuos_aleatorios})
            conexoes = [(m,n) for m in individuos_aleatorios 
                        for n in individuos_aleatorios if m != n ]
            self.G_s.add_edges_from(conexoes)

        nx.set_edge_attributes(self.G_s, 1, 'weight')

        self.pop_tx_transmissao_s = \
            np.array([self.beta_s / (1+self.G_s.degree(i)) 
                      for i in self.G_s.nodes])
        attr_transmissao_s = dict([(i, {'taxa de transmissao': self.
                                        pop_tx_transmissao_s[i]}) 
                                   for i in self.G_s.nodes])

        nx.set_node_attributes(self.G_s, attr_transmissao_s)

    def cria_redes(self):
        self.cria_rede_residencial()
        self.cria_rede_social()
        self.redes = [self.G_r, self.G_s]
        self.redes_tx_transmissao= [self.pop_tx_transmissao_r, self.pop_tx_transmissao_s]

        aux = np.array(
            [
                np.sum(
                    self.f_kernel(
                        np.linalg.norm(
                            self.pop_posicoes - self.pop_posicoes[i], axis=1)
                        )
                    ) 
                    for i in range(self.num_pop)
            ]
        )
        self.pop_fator_tx_transmissao_c = self.beta_c / aux
        