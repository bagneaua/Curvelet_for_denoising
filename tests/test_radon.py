from src.utils import radonTransform, inv_radonTransfrom, psnr
import numpy.random as rd

def test_radon():
    tab = rd.rand(256,256)
    radon = radonTransform(tab)
    inv = inv_radonTransfrom(radon)
    assert psnr(tab, inv) > 50
    assert psnr(tab, inv) < 100