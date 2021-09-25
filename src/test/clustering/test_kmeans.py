from src.clustering.k_means import elkan_functions,k_means
from src.utils.utils_vector import distance_matrix
import numpy as np


class Test_elkan_functions:

    def test_init_bounds_liga_um_ponto_e_um_centro_liga(self):
        x = np.array([[0,0]])
        y = np.array([[1,0]])
        distance = distance_matrix(y,y)/2 #type: ignore
        #Ponto 0,centro 0, distancia 1
        assert np.all(elkan_functions.init_bounds_elkan(x,y,distance)==np.array([0,0,1.]))

    def test_init_bounds_liga_um_ponto_e_dois_centro_liga(self):
        x = np.array([[0,0]])
        y = np.array([[2,1],[1,1]])
        distance = distance_matrix(y,y)/2 #type: ignore
        #Ponto 0,centro 1, distancia 1.4
        assert np.all(np.round(elkan_functions.init_bounds_elkan(x,y,distance),1)==np.array([0,1,1.4]))

    def test_init_bounds_liga_multiplos_pontos_e_multiplos_centro(self):
        x = np.array([[0,0],[2,4],[-1,-3]])
        y = np.array([[2,1],[1,1],[-1,0]])
        distance = distance_matrix(y,y)/2 #type: ignore
        #Ponto 0,centro 1, distancia 1.4
        print(elkan_functions.init_bounds_elkan(x,y,distance))
        assert np.all(elkan_functions.init_bounds_elkan(x,y,distance)==[(0,2,1.0),
                                                                        (1,0,3.0),
                                                                        (2,2,3.0)])

