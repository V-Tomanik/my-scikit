import numpy as np


def distance_vector(x,y):
    """
    x (np.array): ponto x
    y (np.array): ponto y
    """
    xx= np.dot(x,x)
    yy= np.dot(y,y)
    distance = np.sqrt(xx - 2*np.dot(x,y) + yy)
    return distance





if __name__ == '__main__':
    x = np.array([1,1])
    y = np.array([2,2])
    print(distance_vector(x,y))
    
