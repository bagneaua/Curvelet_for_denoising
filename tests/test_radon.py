from src.utils import radonTransform, inv_radonTransfrom, psnr
import numpy.random as rd

def test_radon():
    tab = rd.rand(256,256)
    tab *= 255
    radon = radonTransform(tab)
    inv = inv_radonTransfrom(radon)
    assert psnr(tab, inv) > 30
    assert psnr(tab, inv) < 50