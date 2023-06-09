#%%
import numpy as np

#%%
def BE(M, Yo, RYe):
    """
    Compute selection and complementarity effects

    Parameters
    ----------
    M : Numpy array.
        n-array of values of monoculture production

    Yo : Numpy array.
    	 n-array of values of policulture production

    RYe: Numpy array
         n-array of values of expected relative yield

    Returns
    -------
    3 float tuple with selection effects, complementarity effects and net effects
    """
    Yot = np.sum(Yo)
    RYo = Yo/M
    Ye = RYe * M
    Yet = np.sum(Ye)
    delta_Y = Yot - Yet
    delta_RY = RYo - RYe
    N = len(M)
    CE = N*np.mean(delta_RY)*np.mean(M)
    SE = N*np.cov(delta_RY, M, bias=True)[0][1]
    return SE, CE, delta_Y

#%%
def tripartite_partition(M, Yo, RYe):
    """
    Compute tripartite partition (Fox, 2005; 10.1111/j.1461-0248.2005.00795.x)

    Parameters
    ----------
    M : Numpy array.
        n-array of values of monoculture production

    Yo : Numpy array.
    	 n-array of values of policulture production

    RYe: Numpy array
         n-array of values of expected relative yield

    Returns
    -------
    3 float tuple with trait-independent complementarity, dominance effect
    and trait-dependent complementarity.
    
    """
        
    Yot = np.sum(Yo)
    RYo = Yo/M
    delta_RY = RYo - RYe
    N = len(M)
    trait_indep_complementarity = N*np.mean(delta_RY)*np.mean(M)
    dominance = N*np.cov(M, (RYo / RYo.sum() ) - RYe, bias=True)[0][1]
    trait_dep_complementarity = N*np.cov(M, RYo - (RYo / RYo.sum()), bias=True)[0][1]

    return trait_indep_complementarity, dominance, trait_dep_complementarity

#%%
def Y2_in_LER_equal(M1,M2,Y1,LER):
    '''
    Get intercrop production value of crop 2 (Y2) that gives a specificed LER value given Y1, M1 and M2 values.

    Parameters
    ---------
    M1 : float
        production in monoculture of crop 1
    
    M2 : float
        production in monoculture of crop 2
    
    Y1 : float
        production in intercrop of crop 1

    LER: float
       desired LER value

    Returns
    -------
    corresponding intercrop production value of crop 2 given M1, M2, Y1 and LER
    '''
    return (LER*M2) - ((M2/M1)*Y1)

#%%
def LER(M,Yo,RYe):
    '''
    Get LER.
    '''
    return np.array([y/(2*m*r) for m,y,r in zip(M,Yo,RYe)]).sum()
