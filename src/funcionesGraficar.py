#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
from funcionesCalculos import BE, tripartite_partition, LER

#%%
def plot_yield(M, Yo, ax=None, max_y=None):
    """
    Line plot of mean monoculte yield and observed mix yield.

    Parameters
    ----------
    M : Numpy array.
        n-array of values of monoculture production

    Yo : Numpy array.
    	 n-array of values of policulture production

    ax: Matplotlib Axes object, optional
        Draw the graph in specified Matplotlib axes.

    max_y: float or integer, optional
        Maximum y value.

    Returns
    -------
    Matplotlib AxesSubplot object.
    """
    ax = ax or plt.gca()
    point_size = 25
    line_width = .75
    for m in M:
        ax.scatter(['mono']*len(M), M, s=point_size, color='k', linewidth=line_width, marker='s')
        ax.scatter(['poli'], [Yo.sum()], s=point_size, color='k')
        ax.scatter(['mono'], [M.mean()], s=point_size, color='k', facecolors='None', edgecolors='black')
        ax.plot(['mono','poli'], [M.mean(), Yo.sum()], color='k', linestyle='-', linewidth=line_width)
        ax.axhline(M.max(), color='k', linestyle=':', linewidth=line_width)
        if max_y == None:
            max_y = max([max([m for m in M]), Yo.sum()]) + M.min()/2
        ax.set_ylim(0, max_y)
        ax.set_xlim(-0.25, 1.25)
    ax.set_title("Producción")        
    return ax

#%%
def plot_effects(M,
                 Yo,
                 RYe,
                 effects,
                 ax=None,
                 max_norm=1,
                 cmap='bwr_r',
                 max_y=None,
                 min_y=None):
    """
    Bar plot of selected biodiversity effect.

    Parameters
    ----------
    M : Numpy array.
        n-array of values of monoculture production

    Yo : Numpy array.
    	 n-array of values of policulture production

    RYe: Numpy array
         n-array of values of expected relative yield

    effects: sting or array of strings
        A string or array of strings indicatating the effects to plot.
        Effects can be: "NE" (net effect), "SE" (selection effect),
        "CE" (complementarity effect), "tic" (trait-independent complementarity),
        "dom" (dominance effect), "tdc" (trait-dependent complementarity)
    
    ax: Matplotlib Axes object, optional
        Draw the graph in specified Matplotlib axes.

    max_norm: float or integer, optional (default=1)
        Color map normalization maximum value

   cmap: Matplotlib colormap, optional (default='bwr_r')
        Colormap for mapping intensities of effects

    max_y: float or integer, optional
        Maximum y value.

    min_y: float or integer, optional
        Maximum y value.
    
    Returns
    -------
    Matplotlib AxesSubplot object.
    """
    ax = ax or plt.gca()
    line_width=.75
    cmapa = matplotlib.cm.get_cmap(cmap)
    norm = colors.TwoSlopeNorm(vmin=-max_norm, vcenter=0, vmax=max_norm)
    SE, CE, NE = BE(M, Yo, RYe)
    tic, dom, tdc = tripartite_partition(M, Yo, RYe)
    effects_values = list()
    color_list = list()
    for effect in effects:
        match effect:
            case "NE":
                effects_values.append(NE)
                color_list.append('k')
            case "SE":
                effects_values.append(SE)
                color_list.append(cmapa(norm(SE)))
            case "CE":
                effects_values.append(CE)
                color_list.append(cmapa(norm(CE)))
            case "tic":
                effects_values.append(tic)
                color_list.append(cmapa(norm(tic)))
            case "dom":
                effects_values.append(dom)
                color_list.append(cmapa(norm(dom)))
            case "tdc":
                effects_values.append(tdc)
                color_list.append(cmapa(norm(tdc)))
    ax.bar(effects, effects_values, color=color_list, edgecolor='k')
    ax.axhline(0, color='k', linestyle='-', linewidth=line_width)
    if max_y and min_y:
        ax.set_ylim(min_y, max_y)
    ax.set_title("Tamaño de efecto")
    return ax

def plot_effect_LER(M,
                    RYe,
                    effect,
                    max_x,
                    max_y,
                    ax=None,
                    Yo=None,
                    plot_guide_lines=False,
                    plot_overyield=False,
                    plot_expected_yield=True,
                    plot_guide_points=True,
                    cmap='bwr_r',
                    matrix_resolution = 50):
    """
    LER plot with selected biodiversity effect.

    Parameters
    ----------
    M : Numpy array.
        n-array of values of monoculture production

    RYe: Numpy array
         n-array of values of expected relative yield

    effects: sting or array of strings
        A string or array of strings indicatating the effects to plot.
        Effects can be: "NE" (net effect), "SE" (selection effect),
        "CE" (complementarity effect), "tic" (trait-independent complementarity),
        "dom" (dominance effect), "tdc" (trait-dependent complementarity)

    max_x: float or integer
        Maximum x-cordinate value (maximum cultivar 1 yield to plot)

    max_y: float or integer
        Maximum y-cordinate value (maximum cultivar 2 yield to plot)
    
    ax: Matplotlib Axes object, optional
        Draw the graph in specified Matplotlib axes.

    Yo : Numpy array.
    	 n-array of values of policulture production

    plot_guide_lines: boolean (default=False)
        display guide lines of values of M and RYe

    plot_overyield: boolean (default=False)
        display line of overyielding

    plot_expected_yield: boolean (default=True)
        display point of expected yield

    plot_guide_points: boolean (default=True)
        display points of M

    cmap: Matplotlib colormap, optional (default='bwr_r')
        Colormap for mapping intensities of effects

    matrix_resolution_size: floar or integer (default=50)
        number of pixels used for the longest axis for calulate
        the effect matrix.
    
    Returns
    -------
    Matplotlib AxesSubplot object.
    """
    if max_x >= max_y:
        y_size = round(max_y*matrix_resolution/max_x)
        x_values = np.linspace(0,max_x,matrix_resolution)
        y_values = np.linspace(0,max_y,y_size)
        matrix = np.empty((y_size,matrix_resolution))
    else:
        x_size = round(max_x*matrix_resolution/max_y)
        x_values = np.linspace(0,max_x,x_size)
        y_values = np.linspace(0,max_y,matrix_resolution)
        matrix = np.empty((matrix_resolution,x_size))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            yield_values = np.array([x_values[j],y_values[i]])
            SE, CE, NE = BE(M, yield_values, RYe)
            tic, dom, tdc = tripartite_partition(M, yield_values, RYe)
            if effect == "NE":
                matrix[i][j] = NE
            if effect == "SE":
                matrix[i][j] = SE
            if effect == "CE":
                matrix[i][j] = CE
            if effect == "tic":
                matrix[i][j] = tic
            if effect == "dom":
                matrix[i][j] = dom
            if effect == "tdc":
                matrix[i][j] = tdc
    ax = ax or plt.gca()
    point_size = 25
    line_width = 0.75
    # line LER = 1.0
    ax.plot((M[0]*RYe.sum(),0),(0,M[1]*RYe.sum()), color='k', linewidth=line_width)
    # effect color matrix
    norm = colors.TwoSlopeNorm(vmin=-max_x, vcenter=0, vmax=max_x)
    ax.pcolor(x_values, y_values, matrix, cmap=cmap,norm=norm)
    ax.set_xlim(0,max_x)
    ax.set_ylim(0,max_y)
    ax.set_xlabel(r'$Y_1$')
    ax.set_ylabel(r'$Y_2$')
    ax.set_xticks([M[0]])
    ax.set_yticks([M[1]])
    # point observed yield
    if Yo.any():
        ax.scatter(Yo[0], Yo[1], s=point_size, color='k')
    if plot_guide_lines:
        # expected yield line guides
        ax.axhline(y=RYe[1]*M[1], color='k',linestyle='-',linewidth=line_width,alpha=0.2)
        ax.axvline(x=RYe[0]*M[0], color='k',linestyle='-',linewidth=line_width,alpha=0.2)
        # monoculture production lines
        ax.axhline(y=M[1], color='k',linestyle='-',linewidth=line_width,alpha=0.2)
        ax.axvline(x=M[0], color='k',linestyle='-',linewidth=line_width,alpha=0.2)
    if plot_guide_points:
        # monoculture production points
        ax.scatter(M[0],0, s=point_size, marker='s', color='k', linewidth=line_width)
        ax.scatter(0,M[1], s=point_size, marker='s', color='k', linewidth=line_width)
    if plot_overyield:
        max_M_index = np.argmax(M)
        # overproduction line
        ax.plot((0,M[max_M_index]),(M[max_M_index],0), color='k', linewidth=line_width, linestyle=':')
    if plot_expected_yield:
        ax.scatter(RYe[0]*M[0],RYe[1]*M[1], s=point_size, edgecolors='k', facecolors='None', linewidth=line_width)
    ax.set_title(effect)
    # ax.set_aspect('equal','box')
    return ax

#%%

def plot_CE_SE_effect_yield(M,
                            Yos,
                            RYe,
                            max_x,
                            max_y,
                            max_effect=None,
                            min_effect=None,
                            cmap='bwr_r'):
    """
    Panel with plots of LER with complementarity effect,
    LER with selection effect, barplot of effects and yield plot

    Parameters:
    ----------

    M : Numpy array.
        n-array of values of monoculture production

    Yos: Numpy array.
        n-array of values of observed intercrop production
    
    RYe: Numpy array.
         n-array of values of expected relative yield

    max_x: float or integer
        Maximum x-cordinate value (maximum cultivar 1 yield to plot)

    max_y: float or integer
        Maximum y-cordinate value (maximum cultivar 2 yield to plot)

    max_effect: float or integer
        Maximum value to use in effect size bar plot (used to set same limits
    when wanting to compare multiple plots)

    min_effect: float or integer
        Minimum value to use in effect size bar plot (used to set same limits
    when wanting to compare multiple plots)

    cmap: Matplotlib colormap, optional (default='bwr_r')
        Colormap for mapping intensities of effects

    """

    # se calculan estos valores para que todas las gráficas usen
    # los mismos límites
    if max_effect == None:
        max_effect = max([max(BE(M,Yo,RYe)) for Yo in Yos])
    if min_effect == None:
        min_effect = min([min(BE(M,Yo,RYe)) for Yo in Yos])
    max_yield = max([Yo.sum() for Yo in Yos])
    min_yield = min(M)

    cols = 5
    rows = len(Yos)
    layout = np.arange(cols*rows).reshape(rows,cols)

    fig, axs = plt.subplot_mosaic(layout,
                                  layout="constrained",
                                  figsize=(9, rows*1.8),
                                  width_ratios=[0.2,0.2,0.2,0.2,0.2])

    for i in range(len(Yos)):
        Yo = Yos[i]
        LERi = LER(M,Yo,RYe)
        axs[i*cols].text(0.1,0.5, '$LER=$%.2f \n $Y_{O1}=$%.1f \n $Y_{O2}=$%.1f' %(LERi, Yo[0], Yo[1] ), dict(size=12))
        axs[i*cols].axis("off")
        plot_effect_LER(M,RYe,"CE",max_x,max_y,Yo=Yo,ax=axs[i*cols + 1], cmap=cmap,  plot_overyield=True)
        plot_effect_LER(M,RYe,"SE",max_x,max_y,Yo=Yo,ax=axs[i*cols + 2], cmap=cmap, plot_overyield=True)
        plot_effects(M, Yo, RYe, ["NE","CE","SE"], ax=axs[i*cols + 3], max_norm=max_x, cmap=cmap, max_y=max_effect+1, min_y = min_effect-1)
        plot_yield(M, Yo, ax=axs[i*cols + 4], max_y=max_yield+min_yield)
    plt.suptitle("$RY_{E1}$ = %.1f     $RY_{E2}$ = %.1f" %(RYe[0],RYe[1]))
    for i in range(cols*rows):
        if i >= 3:
            axs[i].set_title("")
        if i <= cols*rows - 6:
            axs[i].set_xlabel("")
            axs[i].tick_params('x',labelbottom=False)
        if i%cols - 3 == 0 :
            axs[i].set_ylabel("tamaño de efecto")
        if i%cols - 4 == 0 :
            axs[i].set_ylabel("producción")
    fig.tight_layout(pad=1.2)
    return fig
