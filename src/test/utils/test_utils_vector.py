from src.utils import utils_vector as uv
import numpy as np


class TestDistance_vector:
    
    def test_distance_two_dimensions_euclidian(self):
        x = np.array([1,1])
        y = np.array([2,2])
        assert round(uv.euclidian_distance(x,y),2) == 1.41
    
    def test_distance_tree_dimensions_euclidian(self):
        x = np.array([1,1,1])
        y = np.array([2,2,2])
        assert round(uv.euclidian_distance(x,y),2) == 1.73
    
    def test_distance_two_dimensions_minkowski(self):
        x = np.array([1,1])
        y = np.array([2,2])
        assert np.round(uv.minkowski_distance(x,y,p=2),2) == 1.41
        assert np.round(uv.minkowski_distance(x,y,p=1),2) == 2
    
    def test_distance_tree_dimensions_minkowski(self):
        x = np.array([1,1,1])
        y = np.array([2,2,2])
        assert np.round(uv.minkowski_distance(x,y),2) == 1.73
        assert np.round(uv.minkowski_distance(x,y,p=1),2) == 3
    
    def test_matrix_two_dimensions_distance(self):
        x = np.array([[0,0],[0,1],[1,0],[3,4]])
        y = np.array([[0,0],[0,1],[1,0],[3,4]])
        assert np.all(np.round(uv.distance_matrix(x,y),1)==np.array([[0.,1.,1.,5.],
                                                                     [1.,0.,1.4,4.2],
                                                                     [1.,1.4,0.,4.5],
                                                                     [5.,4.2,4.5,0.]]))

    def test_matrix_tree_dimensions_distance(self):
        x = np.array([[0,0,0],[0,1,0],[1,1,1],[3,4,1]])
        y = np.array([[0,0,0],[0,1,0],[1,1,1],[3,4,1]])
        assert np.all(np.round(uv.distance_matrix(x,y),1)==np.array([[0.,1.,1.7,5.1],
                                                                     [1.,0.,1.4,4.4],
                                                                     [1.7,1.4,0.,3.6],
                                                                     [5.1,4.4,3.6,0.]]))
