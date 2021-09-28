from src.clustering.k_means import elkan_functions,k_means
from src.utils.utils_vector import distance_matrix
import numpy as np


class Test_elkan_functions:

    def test_init_bounds_liga_um_ponto_e_um_centro(self):
        x = np.array([[0,0]])
        y = np.array([[1,0]])
        distance = distance_matrix(y,y)/2 #type: ignore
        #Ponto 0,centro 0, distancia 1
        assert np.all(elkan_functions.init_bounds_elkan(x,y,distance)==[(0,0,1.)])

    def test_init_bounds_liga_um_ponto_e_dois_centro(self):
        x = np.array([[0,0]])
        y = np.array([[2,1],[1,1]])
        distance = distance_matrix(y,y)/2 #type: ignore
        #Ponto 0,centro 1, distancia 1.4
        assert np.all(np.round(elkan_functions.init_bounds_elkan(x,y,distance),1)==[(0,1,1.4)])

    def test_init_bounds_liga_multiplos_pontos_e_multiplos_centro(self):
        x = np.array([[0,0],[2,4],[-1,-3]])
        y = np.array([[2,1],[1,1],[-1,0]])
        distance = distance_matrix(y,y)/2 #type: ignore
        #Ponto 0,centro 1, distancia 1.4
        assert np.all(elkan_functions.init_bounds_elkan(x,y,distance)==[(0,2,1.0),
                                                                        (1,0,3.0),
                                                                        (2,2,3.0)])
    def test_init_iter_um_ponto_e_dois_centros(self):
        x = np.array([[0.,0.]])
        y = np.array([[2.,1.],[1.,1.]])
        distance = distance_matrix(y,y)/2 #type: ignore

        points,center,hist_bounds,d_distance = elkan_functions.init_iter(points=x,centers=y)
        assert np.all(x==points)
        assert np.all(center== np.array([[2.,1.],[0.,0.]]))
        assert np.all(d_distance==distance)
        assert np.all(np.round(hist_bounds,1) ==  [(0,1,1.4)])

    def test_init_iter_multiplos_pontos_e_multiplos_centros(self):
        x = np.array([[0,0],[2,4],[-1,-3]])
        y = np.array([[2,1],[1,1],[-1,0]])
        distance = distance_matrix(y,y)/2 #type: ignore

        points,center,hist_bounds,d_distance = elkan_functions.init_iter(points=x,centers=y)
        assert np.all(points==np.array([[0.,0.],[2.,4.],[-1.,-3.]]))
        assert np.all(center==np.array([[2.,4.],[1.,1.],[-0.5,-1.5]]))
        assert np.all(d_distance==distance)
        assert np.all(np.round(hist_bounds,1) ==    [(0,2,1.0),
                                                    (1,0,3.0),
                                                    (2,2,3.0)])

    def test_init_iter_um_ponto_e_dois_centros_3d(self):
        pass

    def test_init_iter_um_ponto_e_dois_centros_4d(self):
        pass

    def test_update_centers_um_centro(self):
        center = np.array([[0,0]])
        point = np.array([[1,0]])
        bound = [[(0,0,1)]]
        elkan_functions.update_centers(X=point,centers=center,bounds=bound)
        assert np.all(center == np.array([[1.,0.]]))

    def test_update_centers_multiplos_centros(self):
        center = np.array([[0.,0.],[2.,1.],[-10.,-5.],[0.,0.],[1.,1.]])
        point = np.array([[1,0],[-1,0],[3,1],[-5,-6],[0,1],[0,-1],[1,2],[3,3]])
        bound = [[(0,0,1),(1,0,1.41),(2,1,1),(3,2,5.09),(4,3,1),(5,3,1),(6,4,1),(7,4,2.82)]]
        elkan_functions.update_centers(X=point,centers=center,bounds=bound)
        assert np.all(center == np.array([[0.,0.],[3.,1.],[-5.,-6.],[0.,0.],[2.,2.5]]))

