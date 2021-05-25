import numpy as np
import numpy.typing as npt

class elkan_functions():

    @staticmethod
    def distance_between_centers(centers):
        """
        dist(x,y) = sqrt( x**2 - 2xy + y**2)
        """
        xx = np.dot(centers,centers)
        return xx


class k_means():
    """
    Para a criacao desse modelo, foram usados como base
    a implementacao do scikitlearn e os seguintes papers
    Kmeans Elkan:  https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf


    n_cluster(int): Numero de cluster a serem tradados
    points (matrix:n_points,coordenadas,cluster)
    centers (number_centroid,array(coordenadas))

    """
    def __init__(self,n_cluster):

        self.n_cluster = n_cluster
        self.points = None
        self.centroids = None

    def _init_centroids(self,dimension,x_min,x_max) -> list:
        list_centroids= []
        for _ in range(self.n_cluster):
            list_centroids.append(np.random.uniform(x_min,x_max,dimension))
        return list_centroids


    def elkan_kmeans_iter(self,points,centers):
        pass


    def fit(self,X) -> None:

        x_min = np.amin(X,axis=0)
        x_max = np.amax(X,axis=0)

        n_dimension = X.shape[1]
        self.centroids = np.stack(self._init_centroids(n_dimension,x_min,x_max),axis=0)


if __name__ == '__main__':
    x = k_means(5)
    dados = np.random.uniform(0,100,(3,2))
    print(f'dados {dados}')
    print(x.fit(dados))

