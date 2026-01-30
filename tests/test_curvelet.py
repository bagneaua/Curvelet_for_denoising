import numpy.random as rd

from src.utils import psnr
from src.curvelet import curveletTransform, inv_curveletTransform 

def test_curvelet():
    tab = rd.rand(32,32)
    tab *= 255
    J = 2
    B = 16
    c, curvelet = curveletTransform(tab, J, B)
    inv = inv_curveletTransform(c, curvelet, J, B, tab.shape)
    assert psnr(tab, inv) > 100
    