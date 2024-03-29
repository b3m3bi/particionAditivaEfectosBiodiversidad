#%%
import numpy as np
import matplotlib.pyplot as plt
from funcionesGraficar import plot_yield, plot_effect_LER, plot_effects, plot_CE_SE_effect_yield
from funcionesCalculos import BE, LER

#%%
# Parámetros de estilo de las gráficas
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)
plt.rc('axes', titlesize=12)
plt.rc('figure', titlesize=15)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
cmap = 'PuOr'
#%%
# Se hacen muchos paneles usando un gradiente de parámetos
RYes = np.array([[0.4,0.6],[0.5,0.5],[0.6,0.4]])
# RYes = np.array([[0.5,0.5]])
Ms = np.array([[10,6],[15,6],[20,6]])
# Ms = np.array([[15,6]])
Yos = np.array([[14,2.2],[9,4.2],[5,5.8]])

max_effect = max([max(BE(M,Yo,RYe)) for Yo in Yos for RYe in RYes for M in Ms])
min_effect = min([min(BE(M,Yo,RYe)) for Yo in Yos for RYe in RYes for M in Ms])


max_x, max_y = np.max(Ms) + 2, np.max(Ms) + 2

for i in range(len(RYes)):
    for j in range(len(Ms)):
              plot_CE_SE_effect_yield(Ms[j],Yos,RYes[i],max_x,max_y, max_effect=max_effect,min_effect=min_effect,cmap=cmap)
              # plt.show()
              plt.savefig('../results/exploracion-BE-merge-%d-%d.png' %(i,j), dpi=100)

#%%
# # Panel que compara todos los effectos Loreau y Héctor + Fox
# M = np.array([500,250])
# RYe = np.array([0.6,0.4])
# max_x, max_y = 600,350

# Yo = np.array([390,130])

# fig, axs = plt.subplots(2,3, figsize=(7,5))
# plot_effect_LER(M,RYe,"CE",max_x,max_y,Yo=Yo,ax=axs[0][0], cmap=cmap)
# plot_effect_LER(M,RYe,"SE",max_x,max_y,Yo=Yo,ax=axs[0][1], cmap=cmap)
# plot_effect_LER(M,RYe,"tic",max_x,max_y,Yo=Yo,ax=axs[1][0], cmap=cmap)
# plot_effect_LER(M,RYe,"dom",max_x,max_y,Yo=Yo,ax=axs[1][1], cmap=cmap)
# plot_effect_LER(M,RYe,"tdc",max_x,max_y,Yo=Yo,ax=axs[1][2], cmap=cmap)

# # plot_yield(M, Yo, ax=axs[1][0])
# plot_effects(M, Yo, RYe, ax=axs[0][2], effects=["NE","CE","SE","tic","dom","tdc"], max_norm=max_x, cmap=cmap)
# plt.tight_layout()
# plt.show()
