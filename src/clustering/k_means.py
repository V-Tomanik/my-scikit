import numpy as np
from src.utils.utils_vector import distance_matrix,minkowski_distance


class elkan_functions():

    @staticmethod
    def init_bounds_elkan(X:np.ndarray,centers:np.ndarray,centers_half_distance:np.ndarray) -> list:
        bounds = []
        number_samples = X.shape[0]
        n_cluster = centers.shape[0]

        for i in range(number_samples):
            melhor_cluster = 0 

            #calcula distancia entre ponto e o centro 0
            min_dist = minkowski_distance(X[i],centers[0])

            for j in range(1,n_cluster):
                if min_dist > centers_half_distance[melhor_cluster][j]:  #Lemma1 elkan
                    #caso realmente haja a necessidade, calculamos os valores da distancia 
                    dist = minkowski_distance(X[i],centers[j])
                    if min_dist > dist:
                        min_dist = dist
                        melhor_cluster = j
            bounds.append((i,melhor_cluster,min_dist))
        return bounds


    @staticmethod
    def iter_elkan(X:np.ndarray,centers:np.ndarray,bounds:list,centers_half_distance:np.ndarray) -> list:
        last_iter = bounds[-1]
        number_samples = X.shape[0]
        n_cluster = centers.shape[0]

        for i in range(number_samples):
            current_cluster = last_iter[i][1]
            current_distance = last_iter[i][2]
            future_cluster = None
            future_distance = None 

            for j in range(1,n_cluster):
                if current_distance > centers_half_distance[current_cluster][j]:  #Lemma1 elkan
                    #caso realmente haja a necessidade, calculamos os valores da distancia 
                    dist = minkowski_distance(X[i],centers[j])
                    if current_distance > dist:
                        future_distance = dist
                        future_cluster = j
                    else:
                        raise ValueError("current_distance <= dist")
                else:
                    future_cluster = current_cluster
                    future_distance = current_distance

            bounds.append((i,future_cluster,future_distance))
        return bounds


    @staticmethod
    def update_centers(X,center,bounds) -> np.ndarray:
        return center

class k_means():
    """
    Para a criacao desse modelo, foram usados como base
    a implementacao do scikitlearn e os seguintes papers
    Kmeans Elkan:  https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf


    n_cluster(int): Numero de cluster a serem tradados
    points array:(coordenadas)
    centers array(coordenadas)
    upper_bound: list(list(tuple)) (ponto,centro,distancia) para cada iteração na ordem dos pontos
    
    example:
        points: [[1,2,3],[1,1,1],[2,3,5]]
        centers: [[1,1,1],[2,1,3],[2,3,4]]
        upper_bound:[(0,1,0.3),(0,2,0.2),(1,2,0.5),(1,3,0.1)]
        """
    def __init__(self,n_cluster):

        self.n_cluster = n_cluster
        self.points = None
        self.centroids = None

    def _init_centroids(self,n_clusters,dimension,x_min,x_max):
        list_centroids= []
        for _ in range(n_clusters):
            list_centroids.append(np.random.uniform(x_min,x_max,dimension))
        return np.stack(list_centroids,axis=0)


    def elkan_kmeans_iter(self,points,centers):
        #todo:calcular distancia entre centros

        #loop para cada ponto
            #todo: difinir ponto-centro
            #todo: verificar se distance ponto-centro é menor que 2x a distancia entre os outros centros

        #todo:mudar os centros vendo o centro de massa
        d_matrix = distance_matrix(centers,centers)/2
        x = elkan_functions.init_bounds_elkan(points,centers,d_matrix)
        print(x)


    def fit(self,X) -> None:

        x_min = np.amin(X,axis=0)
        x_max = np.amax(X,axis=0)

        n_dimension = X.shape[1]
        self.centroids = self._init_centroids(self.n_cluster,n_dimension,x_min,x_max)
        self.elkan_kmeans_iter(X,self.centroids)

if __name__ == '__main__':
    x = k_means(5)
    dados = np.random.uniform(0,100,(3,2))
    print(f'dados {dados}')
    x.fit(dados)

