from src.utils import curveletTransform, inv_curveletTransform, psnr
import numpy.random as rd

def test_curvelet():
    tab = rd.rand(32,32)
    J = 2
    B = 16
    c, curvelet = curveletTransform(tab, J, B)
    inv = inv_curveletTransform(c, curvelet, J, B, tab.shape)
    assert psnr(tab, inv) > 50
    