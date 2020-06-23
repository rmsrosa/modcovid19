# Cria a lista de arestas.
#arestas = [('A', 'B'), ('B', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'E'), ('D', 'A'), ('E', 'B')]

# Cria e imprime o grafo.
#grafo = Grafo(arestas, direcionado=True)
#print(grafo.adj)


##############################################################################
# Copyright (c) 2015, Network Science and Engineering Group (NetSE group)) at Kansas State University.
# http://ece.k-state.edu/sunflower_wiki/index.php/Main_Page
#
# W:heman@ksu.eduritten by:
# Heman Shakeri
# All rights reserved.
#
# For details, see https://github.com/scalability-llnl/AutomaDeD
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License (as published by
# the Free Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
# conditions of the GNU General Public License for more details.
##############################################################################
import numpy as np
import networkx as nx
import scipy
import matplotlib.pyplot as plt
import random
import numpy.random as rand
# %matplotlib inline
from scipy.sparse import *
from scipy import *
from scipy.sparse import coo_matrix, bmat
import itertools


# In[38]:

def NeighborhoodData ( N , L1 , L2, W):
    """
    A DB that gives the adjacent nodes (not necessary neighbors). For directed graphs we need NNeighborhoodData too.
    (Heman)
    """    
    junk = np.sort ( L1 ) 
    index = np.argsort( L1 )
    

    NeighVec = L2[index]
    NeighWeight = W[index]

#     junk = np.sort ( dummy1 ) 
#     index = np.argsort( dummy1 )
#     for i in index:
#         NeighVec.extend([dummy2[i]])
#         NeighWeight.extend([dummy3[i]])
          
    l = len( junk );     d = np.zeros ( N , dtype=int32) 
    I1 = np.zeros ( N , dtype=int32) ;
#     I1 = np.zeros ( N ) ;     I2 = np.zeros ( N ) 
    I1 = -np.ones ( N , dtype=int64) #starts from -1, since the first edge is zero
    i = 0 
    while i+1 < l:  #i starts from zero
        node = junk[i]
        I1[node] = i #link number, starts from 0
        while junk[i + 1] == junk[i]:
            d[node] = d[node] + 1 ;
            i += 1
            if i+1 == l:
                break
        i += 1
    if i+1 == l:
        node = junk[i]; I1[node] = i; d[node] = 0 
    I2 = I1 + d 
    Temp1 = np.subtract(I2,I1)
    Temp2 = [int(I1[i]!=0) for i in range(len(I1)) ]
#     d = np.sum(Temp1, Temp2 )    
    return NeighVec, I1, I2, d, NeighWeight   
#--------------------------------------------------------
def NNeighborhoodData ( N , L1 , L2, W): 
    """
    A DB that gives the adjacent nodes (not necessary neighbors). Useful only for directed graphs.
    (Heman)
    """
#     NNeighVec = []; NI1 = [] ; NI2 = [] ; dummy1 = L1; dummy2 = L2 ; dummy3 = W
#     NNeighWeight = []
    
#     junk = np.sort ( dummy2 ) 
#     index = np.argsort( dummy2 )
#     #instead of the following for loop: NNeighVec = dummy1[index]
#     for i in index:
#         NNeighVec.extend([dummy1[i]])
#         NNeighWeight.extend([dummy3[i]])
   
    
    junk = np.sort ( L2 ) 
    index = np.argsort( L2 )

    NNeighVec = L1[index]
    NNeighWeight = W[index]
          
    l = len( junk );     Nd = np.zeros ( N , dtype=int32) 
#     NI1 = np.zeros ( N , dtype=int32)
    NI1 = -np.ones ( N , dtype=int64) 
    i = 0 
    I1 = [] ; I2 = [] ;
    while i+1 < l:  #i starts from zero
        node = junk[i]
        NI1[node] = i #link number, starts from 0
        while junk[i + 1] == junk[i]:
            Nd[node] = Nd[node] + 1 ;
            i += 1
            if i+1 == l:
                break
        i += 1
    if i+1 == l:
        node = junk[i]; NI1[node] = i; Nd[node] = 0 
    NI2 = NI1 + Nd 
    Temp1 = np.subtract(NI2, NI1)
    Temp2 = [int(NI1[i]!=0) for i in range(len(NI1)) ]
#     d = np.sum(Temp1, Temp2 )    
    return NNeighVec, NI1, NI2, Nd, NNeighWeight   


# In[48]:

def EIG1(G):
    adj=nx.to_scipy_sparse_matrix(G)
    k=adj.sum(axis=1);
    k=k/float(k.sum())
    err = 1; lambda1 = 0
    while err>1e-3:
        k = adj.dot(k)
#         k = np.dot(adj,k)
        temp = k.sum()
        err = temp - lambda1
        lambda1 = temp
        k = k/lambda1
        v1 = k
    return lambda1, v1
#----------------------------
def Initial_Cond_Gen(N, J, NJ, x0):
    """
    J = initial state for NJ number of whole population N
    Example :   x0 = np.zeros(N, dtype = int32)
                Initial_Cond_Gen(10, Para[1][0], 2, x0)
    """
    if sum(NJ) > N:
        return 'Oops! Initial infection is more than the total population'
    else:
        temp = np.random.permutation(N); nj=temp[0:sum(NJ)]
        for i in range(len(nj)):
            x0[nj[i]] = J
    return x0
#----------------------------
def rnd_draw(p):
    """
    To draw a sample using a probability distribution.
    """
    a = [0]
    a = np.append(a, np.cumsum(p[0:-1]))/np.sum(p)
    b = cumsum(p)/np.sum(p)
    toss = rand()
    k = np.intersect1d(np.nonzero(a<toss)[0], np.nonzero(b>=toss)[0])
    return k


# In[49]:

def MyNet(G, weight=None):
    
    """
    MyNet(G, weight='weight')
    """
    G_adj = nx.to_scipy_sparse_matrix(G, weight=weight)
    cx = G_adj.tocoo()  
    L2 = cx.row
    L1 = cx.col
    W = cx.data
    N = G.number_of_nodes()
#     adj = [L1, L2, W, N]
    NeighVec, I1, I2, d, NeighWeight = NeighborhoodData ( N , L1 , L2, W)
    
    if nx.is_directed(G):
#     if True:
        NNeighVec, NI1, NI2, Nd, NNeighWeight = NNeighborhoodData ( N , L1 , L2, W) #ver2
    
#     Net = [NeighVec, I1, I2, d, adj, NeighWeight]
        Net = [NeighVec, I1, I2, d, NeighWeight, NNeighVec, NI1, NI2, NNeighWeight]#ver2
    else:
        Net = [NeighVec, I1, I2, d, NeighWeight]#ver2
    return Net


# In[50]:

def NetCmbn(NetSet):    
    """
    Combine different network layers data. This function is used for directed networks.
    """
   
    if len(NetSet[0])>5: #Means it is directed
        Neigh = []; I1 = []; I2 = []; d = [];  NeighW = []; NNeigh = []; NI1 = []; NI2 = []; NNeighW = []
        #print(len(NetSet))
        for l in range(len(NetSet)):
            #print(l)
            Neigh.append(NetSet[l][0]) #each layer append as a seperate: Neigh = [[Neigh_L1],[Neigh_L2]]
            I1.append(NetSet[l][1])    #I1 and I2 into each row of the new I1 and I2
            I2.append(NetSet[l][2])
            d.append(NetSet[l][3])
            NeighW.append(NetSet[l][4])
            NNeigh.append(NetSet[l][5]) #ver2
            NI1.append(NetSet[l][6])    #ver2
            NI2.append(NetSet[l][7])
            NNeighW.append(NetSet[l][8])
        Net = [Neigh,I1,I2,d, NeighW, NNeigh, NI1, NI2, NNeighW]
    else:
        Neigh = []; I1 = []; I2 = []; d = []; adj = []; NeighW = []
        for l in range(len(NetSet)):
            Neigh.append(NetSet[l][0]) #each layer append as a seperate: Neigh = [[Neigh_L1],[Neigh_L2]]
            I1.append(NetSet[l][1])    #I1 and I2 into each row of the new I1 and I2
            I2.append(NetSet[l][2])
            d.append(NetSet[l][3])
#             adj.append(NetSet[l][4])
            NeighW.append(NetSet[l][4])        
        Net = [Neigh,I1,I2,d, NeighW]
    return Net


# In[51]:

def GEMF_SIM(Para, Net, x0, StopCond, N, Directed = False):
    """
    An event-driven approach to simulate the stochastic process. 
    
    """
    M = Para[0]; q = Para[1]; L = Para[2]; A_d = Para[3]; A_b = Para[4]
    Neigh = Net[0]
    
    I1 = Net[1]
    I2 = Net[2]
    NeighW = Net[4]
    #print(len(I1[0]))
    #print(I1)
    #print(I1[0][378])
    #wait = input("PRESS ENTER TO CONTINUE.")
    n_index = []; j_index = []; i_index = []
    #------------------------------
    bil = np.zeros((M,L))
    for l in range(L):
        bil[:,l] = A_b[l].sum(axis=1) #l'th column is row sum of l'th A_b
    #------------------------------
    bi = np.zeros((M,M,L))
    for i in range(M):
        for l in range(L):
            bi[i, :, l] = A_b[l][i,:]
    #------------------------------
    di = A_d.sum(axis=1) #The rate that we leave compartment i, due to nodal transitions
    #------------------------------
    #X = copy(x0)
    X = x0.astype(int32)#since compartments are just numbers we are using integer types. If 
    
    #------------------------------
    Nq = np.zeros((L,N))
    #------------------------------ver 2
    for n in range(N):
        for l in range(L):
            #print(n)
            #print(l)
            #print(L)
            #print(I1[l][n])
            #print(I2[l][n])
            #print(I1)
            #print(I2)
            Nln = Neigh[l][I1[l][n]:I2[l][n]+1]
            Nq[l][n] = sum((X[Nln]==q[l])*NeighW[l][I1[l][n]:I2[l][n]+1]        )     
    #------------------------------ver2
    Rn = np.zeros(N)
    
    for n in range(N):
#         print 'di[X[n]]: '+str(di[X[n]])
#         print 'Nq[:,n]: '+str(Nq[:,n])
#         print 'bil[X[n],:]: '+str(bil[X[n],:])
#         print 'np.dot(bil[X[n],:],Nq[:,n]): '+str(np.dot(bil[X[n],:],Nq[:,n]))
        Rn[n] = di[X[n]] + np.dot(bil[X[n],:],Nq[:,n])
    R = sum(Rn)
    #------------------------------
    EventNum = StopCond[1]; RunTime= StopCond[1]    
    ts = []
#     #------------------------------
    s=-1; Tf=0 
    if len(Net)>5:
        NNeigh = Net[5]; NI1 = Net[6]; NI2 = Net[7]; NNeighW = Net[8] 
        while Tf < RunTime:
            s +=1
            ts.append(-log( rand() )/R) 
            #print(ts)
            #print(s)
            #------------------------------ver 2
            ns = rnd_draw(Rn)
            
            iss = X[ns]

            js = rnd_draw( np.ravel(A_d[iss,:].T  + np.dot(bi[iss],Nq[:,ns]) ))
            n_index.extend(ns)
            j_index.extend(js)
            i_index.extend(iss)
    #        -------------------- % Updateing ver2
            X[ns] = js
            R -= Rn[ns]
            Rn[ns] = di[js] + np.dot(bil[js,:] , Nq[:,ns])
            R += Rn[ns]       

            infl = (q == js).nonzero()[0]#inf is layers with influencer compartment        
            for l in infl: 
                              
                n1=int(ns)
                Nln = NNeigh[l][NI1[l][n1]:NI2[l][n1]+1] #finding nodes that are adjacent to new infected
                IncreasEff = NNeighW[l][NI1[l][n1]:NI2[l][n1]+1]
                Nq[l][Nln] += IncreasEff #add the new infection weight edges
                k = 0
                for n in Nln:
                    Rn[n] += bil[X[n],l]*IncreasEff[k]
                    R += bil[X[n],l]*IncreasEff[k]
                    k +=1        

            infl2 = (q == iss).nonzero()[0]#infl2 is layers with influencer compartment      
            
    #         print 'inf2: '+str(inf2)
            for l in infl2: #finding influencer compartments
                Nln = NNeigh[int(l)][int(NI1[l][ns]):int(NI2[l][ns])+1] #finding nodes that are adjacent to new infected
                reducEff = NNeighW[int(l)][int(NI1[l][ns]):int(NI2[l][ns])+1]
                Nq[l][Nln] -= reducEff #subtract the new infection weight edges
                k = 0
                for n in Nln:  
                    Rn[n] -= bil[X[n],l]*reducEff[k]         
                    R -= bil[X[n],l]*reducEff[k]
                    k += 1
                #print(R)    
            if R < 1e-6:
                break
            Tf += ts[s]
    else:
        while Tf < RunTime:
            s +=1
            ts.append(-log( rand() )/R)        
            #------------------------------ver 2
            ns = rnd_draw(Rn)
            iss = X[ns]

            js = rnd_draw( np.ravel(A_d[iss,:].T  + np.dot(bi[iss],Nq[:,ns]) ))
            n_index.extend(ns)
            j_index.extend(js)
            i_index.extend(iss)
    #        -------------------- % Updateing ver2
            print(ns)
            print(js)
            X[ns] = js
            #print(ns)
            R -= Rn[ns]
            Rn[ns] = di[js] + np.dot(bil[js,:] , Nq[:,ns])
            R += Rn[ns]       

            infl = (q == js).nonzero()[0]#inf is layers with influencer compartment        
            for l in infl:          
                Nln = Neigh[int(l)][int(I1[l][ns]):int(I2[l][ns])+1] #finding nodes that are adjacent to new infected
                IncreasEff = NeighW[int(l)][int(I1[l][ns]):int(I2[l][ns])+1]
                Nq[l][Nln] += IncreasEff #add the new infection weight edges
                k = 0
                for n in Nln:
                    Rn[n] += bil[X[n],l]*IncreasEff[k]
                    R += bil[X[n],l]*IncreasEff[k]
                    k +=1        

            infl2 = (q == iss).nonzero()[0]#infl2 is layers with influencer compartment      
    #         print 'inf2: '+str(inf2)
            for l in infl2: #finding influencer compartments
                Nln = Neigh[int(l)][int(I1[l][ns]):int(I2[l][ns])+1] #finding nodes that are adjacent to new infected
                reducEff = NeighW[int(l)][int(I1[l][ns]):int(I2[l][ns])+1]
                Nq[l][Nln] -= reducEff #subtract the new infection weight edges
                k = 0
                for n in Nln:  
                    Rn[n] -= bil[X[n],l]*reducEff[k]         
                    R -= bil[X[n],l]*reducEff[k]
                    k += 1
            if R < 1e-6:
                break
            Tf += ts[s] 
            
    return ts, n_index, i_index, j_index


# In[52]:

def Post_Population(x0, M, N, ts, i_index, j_index):

    X0 = np.zeros((M,N))
    for i in range(N):
        X0[int(x0[i])][i] = 1
    T = [0]
    T.extend(np.cumsum(ts))
    StateCount = np.zeros((M,len(ts)+1))
    StateCount[:,0] = X0.sum(axis=1)
    DX = np.zeros(M); DX[i_index[0]] = -1; DX[j_index[0]] = 1
    StateCount[:,1] = StateCount[:,0]+DX
    for k in range(len(ts)):
        DX = np.zeros(M); DX[i_index[k]] = -1; DX[j_index[k]] = 1
        StateCount[:,k+1] = StateCount[:,k] + DX
   
    return T, StateCount


# In[53]:

def Para_SIS(delta,beta):
    M = 2; q = np.array([1]); L = len(q);
    A_d = np.zeros((M,M)); A_d[1][0] = delta
    A_b = []
    for l in range(L):
#         A_b.append(asmatrix(np.zeros((M,M))))
        A_b.append(np.zeros((M,M)))
    A_b[0][0][1] = beta #[l][M][M]
    Para=[M,q,L,A_d,A_b]
    return Para


# In[54]:

def Para_SIR(delta, beta):
    M = 3; q = np.array([1]); L = len(q);
    A_d = np.zeros((M,M));   A_d[1][2] = delta
    A_b = []
    for l in range(L):
        A_b.append(np.zeros((M,M)))
    A_b[0][0][1] = beta #[l][M][M]
    Para=[M,q,L,A_d,A_b]
    return Para


# In[55]:
#         A_b.append(asmatrix(np.zeros((M,M))))
#        A_b.append(np.zeros((M,M)))
#        A_b[l][0][1] = beta[l] #[l][M][M]
def Para_SEIR(delta, beta, Lambda):
    M = 4; q = np.array([2,2,2,2]); L = len(q);
    A_d = np.zeros((M,M));   A_d[1][2] = Lambda; A_d[2][3] = Lambda
    A_b = []
    #A_b.append(np.zeros((M,M)))
    #print(L)
    for l in range(L):
        A_b.append(np.zeros((M,M)))
    #print(A_b)    
    A_b[0][0][1] = beta[0] #[l][M][M]
    A_b[1][0][1] = beta[1] #[l][M][M]
    A_b[2][0][1] = beta[2] #[l][M][M]
    A_b[3][0][1] = beta[3] #[l][M][M]

    Para=[M,q,L,A_d,A_b]
    return Para


# In[56]:

def Para_SAIS_Single(delta, beta, beta_a, kappa):
    M = 3; q = np.array([1]); L = len(q); 
    A_d = np.zeros((M,M)); A_d[1][0] = delta
    A_b = []
    for l in range(L):
        A_b.append(np.zeros((M,M)))
    A_b[0][0][1] = beta #[l][M][M]
    A_b[0][0][2] = kappa 
    A_b[0][2][1] = beta_a 
    Para = [M, q, L, A_d, A_b]
    return Para


# In[57]:

def Para_SAIS(delta, beta, beta_a, kappa, mu):
    M = 3; q = np.array([1,1]); L = len(q); 
    A_d = np.zeros((M,M)); A_d[1][0] = delta
    A_b = []
    for l in range(L):
        A_b.append(np.zeros((M,M)))
    A_b[0][0][1] = beta #[l][M][M]
    A_b[0][0][2] = kappa
    A_b[1][2][1] = beta_a 
    A_b[1][0][2] = mu 
    Para = [M, q, L, A_d, A_b]
    return Para


# In[58]:

def Para_SI1I2S(delta1, delta2, beta1, beta2):
    M = 3; q = np.array([1,2]); L = len(q); 
    A_d = np.zeros((M,M)); A_d[1][0] = delta1; A_d[2][0] = delta2
    A_b = []
    for l in range(L):
        A_b.append(np.zeros((M,M)))
    A_b[0][0][1] = beta1 #[l][M][M]
    A_b[1][0][2] = beta2 #[l][M][M]
    
    Para = [M, q, L, A_d, A_b]
    return Para


# In[59]:

def MonteCarlo(Net, Para, StopCond, Init_inf, M, step, nsim, N, x_init = None):
#     StopCond=['RunTime',500]
#     T_final = 80;
    t_interval = np.arange(0,StopCond[1], step)    
    tsize = int(StopCond[1]/float(step))
    t_interval = np.linspace(0, StopCond[1], num=tsize)
    f = np.zeros(( M, tsize ))
#     nsim = 20; 
    for n in range(nsim):
        x0 = Initial_Cond_Gen(N, Para[1][0], Init_inf, x0 = np.zeros(N, dtype = int32))
        [ts, n_index, i_index, j_index] = GEMF_SIM(Para, Net, x0, StopCond, N)
        [T, StateCount] = Post_Population(x0, M, N, ts, i_index, j_index)
        k=0
        y=np.zeros((M,tsize))
        NewT = T.extend([1000])
        for t in t_interval:
            ind, tr = np.histogram(t,bins = T)
            index = np.nonzero(ind)[0][0]
#             print index
            y[:,k] = StateCount[:, index]/N
            k+=1
        f += y;
    return t_interval, f/nsim


# In[60]:

def Simulation(G, Para, StopCond, Init_inf, nsim, Monte_Carlo = False, step = .1):
    """ -> 
    >>> StopCond = ['RunTime', 20]
    """
    Net = NetCmbn([MyNet(G)])
    N = G.number_of_nodes()
    x0 = np.zeros(N)
    M = Para[0]
    if Monte_Carlo:
#         t_interval, f = MonteCarlo(StopCond, M, T_final, step, nsim, N)
        return MonteCarlo(Net, Para, StopCond, Init_inf, M, step, nsim, N, x_init = x0)
    else:
        x0 = Initial_Cond_Gen(N, Para[1][0], Init_inf, x0)
        ts, n_index, i_index, j_index = GEMF_SIM(Para, Net, x0, StopCond,N)   
#         T, StateCount = Post_Population(x0, M, N, ts, i_index, j_index)
        return Post_Population(x0, M, N, ts, i_index, j_index)
    return T, StateCount
#-------------------------------------------------------------------------
def Sim_vacc(G, Para, C, Init_inf = 3, Num_of_Vacc = None, StopCond = None, nsim = None):
    """
    >>> t_interval, f_pass, f_rnd, f_outDeg, f_Mod = Sim_vacc(H, Para, Init_inf, C_Geo_d, Num_of_Vacc = 30, StopCond = ['RunTime', 30], nsim = 80)
    """
#     StopCond = StopCond
    G_Mod = G_Mod_vaccination(G, C, Num_of_Vacc)
    G_outDeg = G_Out_Deg_vaccination(G, Num_of_Vacc)
    G_rnd = G_rnd_vaccination(G, Num_of_Vacc)
#     G_LEig = G_LeftEig_vaccination(G, Num_of_Vacc)
    t_interval, f_pass = Simulation(G, Para, StopCond, Init_inf, nsim, Monte_Carlo = True )
    t_interval, f_rnd = Simulation(G_rnd, Para, StopCond, Init_inf, nsim, Monte_Carlo = True )
    t_interval, f_outDeg = Simulation(G_outDeg, Para, StopCond, Init_inf, nsim, Monte_Carlo = True )
    t_interval, f_Mod = Simulation(G_Mod, Para, StopCond, Init_inf, nsim, Monte_Carlo = True )
#     t_interval, f_LEig = Simulation(G_LEig, Para, StopCond2, Init_inf = Init_inf, Monte_Carlo = True)
    return t_interval, f_pass, f_rnd, f_outDeg, f_Mod


## Animation

# In[61]:

from matplotlib import animation

def animate_discrete_property_over_graph( g, model, steps, fig, n_index,i_index, j_index, comp, property = None,
                                         color_mapping = None, pos = None, Node_radius = None, **kwords ):
    """Draw a graph and animate the progress of a property over it. The
    property values are converted to colours that are then used to colour
    the nodes.
    """
    x0 = model[0]; n_index = model[1]; i_index = model[2]; j_index = model[3]
    
    # manipulate the axes, since this isn't a data plot
    ax = fig.gca()

    pos
    ax.grid(False)                # no grid
    ax.get_xaxis().set_ticks([])  # no ticks on the axes
    ax.get_yaxis().set_ticks([])
    nx.draw_networkx_edges(g, pos)

    if Node_radius == None:
        Node_radius = .02
        
    # draw the graph, keeping hold of the node markers
    nodeMarkers = []
    for v in g.nodes():
#         circ = plt.Circle(pos[v], radius = 0.02, zorder = 2)   # node markers at top of the z-order
        circ = plt.Circle(pos[v], radius = Node_radius, zorder = 2)   # node markers at top of the z-order
        ax.add_patch(circ)
        nodeMarkers.append({ 'node_key': v, 'marker': circ })

    # initialisation colours the markers according to the current
    # state of the property being tracked
    def colour_nodes():
        for nm in nodeMarkers:
            v = nm['node_key']
            state = g.nodes[v][property]
            c = color_mapping[state]
            marker = nm['marker']
            marker.set(color = c)

    # initialisation coours the markers according to the current
    # state of the property being tracked
    
#     comp = ['S', 'I' ]
    def init_state():
        """Initialise all node in the graph to be susceptible."""
        for i in g.nodes.keys():
            g.nodes[i]['state'] = comp[int(x0[i])]      
        colour_nodes()
        
    # per-frame animation just iterates the model and then colours it
    # to reflect the changed property status of each node
    
    def frame(i):
        changing_node = n_index[i]
        new_comp = j_index[i]
        g.nodes[changing_node]['state'] = comp[new_comp]
        colour_nodes()
    
          
        
        
    # return the animation with the functions etc set up
    return animation.FuncAnimation(fig, frame, init_func = init_state, frames = steps, **kwords)


