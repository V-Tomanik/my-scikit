import numpy as np
from typing import Union

def euclidian_distance(x,y) -> Union[np.ndarray,None]:
    """
    Exemplo
    distance = ((p1x - p2x)² + (p1y - p2y)²)^(0.5)
    x (np.array): ponto x
    y (np.array): ponto y
    """
    xx = np.dot(x,x)
    yy= np.dot(y,y)
    distance = np.sqrt(xx - 2*np.dot(x,y) + yy)
    return distance

def minkowski_distance(x:np.ndarray,y:np.ndarray,p=2) -> Union[np.ndarray,None]:
    """
    Generalização da euclidian e manhattan distance

    Para precisão usar p=2
    """
    if p == 1:
        return np.sum(np.abs(y-x),axis=-1)
    if p == 2:
        return np.sum(np.abs(y-x)**p, axis=-1)**(1./p)
    return None

def distance_matrix(x:np.ndarray,y:np.ndarray) -> Union[np.ndarray,None]:
    """
    Calcula a distancia entre cada um dos pontos informados entre x,y
    """
    x = np.asarray(x)
    k = x.shape[1]
    y = np.asarray(y)
    kk = y.shape [1]
    
    if k != kk:
        raise ValueError("As duas matrizes nao possuem as mesmas dimensaoes")
    #Crio arrays perperdiculares para chegar na matrix de distancia
    return minkowski_distance(x[:,np.newaxis,:],y[np.newaxis,:,:])

if __name__ == '__main__':
    pass
