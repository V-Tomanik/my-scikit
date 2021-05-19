import numpy as np
import numpy.typing as npt

class k_means():
    def __init__(self,n_cluster) -> None:    

        self.n_cluster = n_cluster
        self.cluster_center = None

    def _init_centroid(self,dimension,x_min,x_max) -> npt.ArrayLike:
        centroid = np.random.uniform(x_min,x_max,dimension) 
        return centroid
    def fit(self,X:npt.ArrayLike) -> None:
        list_cluster = []

        x_min = np.amin(X,axis=0)
        x_max = np.amax(X,axis=0)

        n_dimension = x_min.shape[0]
        for _ in range(self.n_cluster):
            cluster=np.empty(n_dimension)
            for i in range(n_dimension):
                cluster[i] = (np.random.uniform(x_min[i],x_max[i]))
            list_cluster.append(cluster)
        self.cluster_center = np.stack(list_cluster)


if __name__ == '__main__':
    x = k_means(5)
    dados = np.random.uniform(0,100,(3,2))
    print(f'dados {dados}')
    x.fit(dados)
    print(x.cluster_center)
