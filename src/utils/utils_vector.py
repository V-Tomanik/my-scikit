import numpy as np


def euclidian_distance(x,y):
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

def minkowski_distance(x,y,p=2):
    """
    Generalização da euclidian e manhattan distance

    Para precisão usar p=2
    """
    if p == 1:
        return np.sum(np.abs(y-x),axis=-1)
    if p == 2:
        return np.sum(np.abs(y-x)**p, axis=-1)**(1./p)
        
def distance_matrix(x,y):
    x = np.asarray(x)
    k = x.shape[1]
    y = np.asarray(y)
    kk = y.shape [1]
    
    if k != kk:
        raise ValueError("As duas matrizes nao possuem as mesmas dimensaoes")
    return minkowski_distance(x[:,np.newaxis,:],y[np.newaxis,:,:])

if __name__ == '__main__':
    #x = np.matrix([[1,1],[2,2],[3,3]])
    #print(distance_matrix(x,x))
    x = np.array([1,1])
    y = np.array([2,2])
    print(np.round(minkowski_distance(x,y,p=1),2))
