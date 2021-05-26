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


