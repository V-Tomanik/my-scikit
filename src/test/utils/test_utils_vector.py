from src.utils import utils_vector as uv
import numpy as np


class TestDistance_vector:
    
    def test_distance_two_dimensions(self):
        x = np.array([1,1])
        y = np.array([2,2])
        assert round(uv.distance_vector(x,y),2) == 1.41
    
    def test_distance_tree_dimensions(self):
        x = np.array([1,1,1])
        y = np.array([2,2,2])
        assert round(uv.distance_vector(x,y),2) == 1.73
    
