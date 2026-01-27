from src.utils import cartesianToPolar, polarToCartesian, psnr
import numpy.random as rd

def test_radon():
    tab = rd.rand(256,256)
    tab *= 255
    polar = cartesianToPolar(tab)
    inv = polarToCartesian(polar)
    assert psnr(tab, inv) > 300
    assert psnr(tab, inv) < 500