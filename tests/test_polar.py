import numpy.random as rd

from src.radon import cartesianToPolar, polarToCartesian
from src.utils import psnr

def test_radon():
    tab = rd.rand(256,256)
    tab *= 255
    polar = cartesianToPolar(tab)
    inv = polarToCartesian(polar)
    assert psnr(tab, inv) > 300
    assert psnr(tab, inv) < 500