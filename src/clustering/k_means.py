import numpy as np
from src.utils.utils_vector import distance_matrix,minkowski_distance
from  matplotlib import pyplot as plt
import pandas as pd

class elkan_functions():
    """
    Classe abriga as funções especificar para o Kmeans de Elkan
    """
    @staticmethod
    def init_bounds_elkan(X:np.ndarray,centers:np.ndarray,centers_half_distance:np.ndarray) -> list:
        """
        Funções retorna o uma lista de array's com os index do  ponto, o index do  melhor cluster e a distancia
        Inicializa o historico
        """
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
                    if min_dist > dist: #type: ignore
                        min_dist = dist
                        melhor_cluster = j
            bounds.append((i,melhor_cluster,min_dist))
        return bounds

    @staticmethod
    def init_iter(points:np.ndarray,centers:np.ndarray):

        hist_bounds =[]
        # Cria uma matriz da distancia entre os centros
        d_matrix = distance_matrix(centers,centers)/2  # type: ignore
        init_bounds  = elkan_functions.init_bounds_elkan(points,centers,d_matrix)
        hist_bounds.append(init_bounds)
        elkan_functions.update_centers(points,centers,hist_bounds) 
        
        return points,centers,hist_bounds,d_matrix


    @staticmethod
    def iter_elkan(X:np.ndarray,centers:np.ndarray,bounds:list,centers_half_distance:np.ndarray) -> list:
        """
        Função para a iteração do Kmeans de elkan, retorna para cada ponto i o centro futuro e a distancia entre os dois
        x: matrix de pontos
        centers: matrix dos centros
        bounds: historico de cada ponto
        centers_half_distance: matriz com a distancia entre os centros
        """
        last_iter = bounds[-1]
        number_samples = X.shape[0]
        n_cluster = centers.shape[0]

        #Para cada ponto
        for i in range(number_samples):
            current_cluster = last_iter[i][1]
            #Distancia do ponto ao centro atualmente
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
                        raise ValueError("current_distance <= distancia do centro. Problema com a matriz dist entre centros")
                else:
                    future_cluster = current_cluster
                    future_distance = current_distance

            bounds.append((i,future_cluster,future_distance))
        return bounds


    @staticmethod
    def update_centers(X:np.ndarray,centers:np.ndarray,bounds:list):
        """
        Faz um update nos centros dos clusters
        bound: Lista com o historio dos pontos [(n_ponto,n_cluster,distancia),....]
        """
        #cria um dicionario dos centros com os pontos dentro
        dict_centers = {k:[] for k in range(centers.shape[0])}
        print(dict_centers)
        last_bound = bounds[-1]
        print(last_bound)
        for point in last_bound:
            #point=(ponto,cluster,distance)
            #Cria uma lista dos centros com seus pontos
            print(point)
            dict_centers[point[1]].append(point[0])
        for center in range(centers.shape[0]):
            #Se o centro tiver algum ponto vamos altera-lo
            if dict_centers[center] != []:
                #Cria um array com pontos (que também é um array da sua posição)
                points_of_array = np.take(X,dict_centers[center],axis=0)
                new_center = np.mean(points_of_array,axis=0)
                centers[center] = new_center

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
        """
        Função de criação dos centroids, cria aleatoriamente os centroids dentro das x_max e x_min
        n_cluster: Número de clusters a serem criados
        dimension: Dimensão do espaço 
        x_min: Minimo valor no eixo
        x_max: Maximo valor no eixo
        """
        list_centroids= []
        for _ in range(n_clusters):
            list_centroids.append(np.random.uniform(x_min,x_max,dimension)) # type: ignore
        return np.stack(list_centroids,axis=0)

    def elkan_kmeans_iter(self,points:np.ndarray,centers:np.ndarray,iterations):
        """
        Função de iteração do Kmeans
        """
        hist_bounds =[]
        #loop para cada ponto
            #todo: difinir ponto-centro
            #todo: verificar se distance ponto-centro é menor que 2x a distancia entre os outros centros
        print(f'centros:{centers}')
        df = pd.DataFrame(centers)
        df2 = pd.DataFrame(points)
        graph = df.plot.scatter(x=0,y=1,color='red')
        df2.plot.scatter(x=0,y=1,color='DarkBlue',ax=graph)
        plt.show()

        #todo:mudar os centros vendo o centro de massa        df = pd.DataFrame(centers)
        points,center,hist_bounds,d_distance = elkan_functions.init_iter(points=points,centers=centers)

        df2 = pd.DataFrame(points)
        graph = df.plot.scatter(x=0,y=1,color='red')
        df2.plot.scatter(x=0,y=1,color='DarkBlue',ax=graph)
        plt.show()
        
        for _ in range(iterations):
            bound = elkan_functions.iter_elkan(points,center,hist_bounds,d_distance)
            hist_bounds.append(bound)
            elkan_functions.update_centers(points,centers,hist_bounds) 
 
            df2 = pd.DataFrame(points)
            graph = df.plot.scatter(x=0,y=1,color='red')
            df2.plot.scatter(x=0,y=1,color='DarkBlue',ax=graph)
            plt.show()
               


    def fit(self,X) -> None:

        x_min = np.amin(X,axis=0)
        x_max = np.amax(X,axis=0)

        n_dimension = X.shape[1]
        self.centroids = self._init_centroids(self.n_cluster,n_dimension,x_min,x_max)
        self.elkan_kmeans_iter(X,self.centroids,3)


if __name__ == '__main__':
    x = k_means(5)
    dados = np.random.uniform(0,100,(50,2)) #type: ignore
    print(f'dados {dados}')
    x.fit(dados)

