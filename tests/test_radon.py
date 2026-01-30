import numpy.random as rd

from src.radon import radonTransform, inv_radonTransfrom
from src.utils import psnr

def test_radon():
    tab = rd.rand(256,256)
    tab *= 255
    radon = radonTransform(tab)
    inv = inv_radonTransfrom(radon)
    assert psnr(tab, inv) > 30
    assert psnr(tab, inv) < 50