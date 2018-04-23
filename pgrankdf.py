# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 23:04:40 2018

@author: Subhajeet
"""

import numpy as np
import pandas as pd
import networkx as nx
#%%
df=pd.read_csv('wiki-Vote.csv',header=None)
#%%
g=nx.DiGraph()
g.add_nodes_from(np.unique(np.array(df[0])))
edge=[tuple(i) for i in df.values]
g.add_edges_from(edge)
#%%
trm=nx.google_matrix(g)
n=min(trm.shape)
p0=np.repeat(1/n,n)
pi=np.matmul(p0,trm)
eps=0.1
#%%
i=1
while np.sum(np.abs(pi-p0))>eps:
    p0=pi
    pi=np.matmul(pi,trm)
    print(i)
    print(pi)
    i=i+1
    if i==20000:
        break
print('The final rank is :',pi)
#%%
#direct command
nx.pagerank(g,alpha=0.88)
#%%
#pagerank function sir's method
def pagerank(g):
    import numpy as np
    import networkx as nx
    trm=nx.google_matrix(g)
    n=min(trm.shape)
    p0=np.repeat(1/n,n)
    pi=np.matmul(p0,trm)
    i=1
    eps=0.00015
    while np.sum(np.abs(pi-p0))>=eps:
        p0=pi
        pi=np.matmul(pi,trm)
        i=i+1
        if i==10000:
            break
    return pi