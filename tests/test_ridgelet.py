import numpy.random as rd

from src.ridgelet import ridgeletTransform, inv_ridgeletTransform
from src.utils import psnr

def test_ridgelet():
    tab = rd.rand(256,256)
    tab *= 255
    ridgelet = ridgeletTransform(tab)
    inv = inv_ridgeletTransform(ridgelet)
    assert psnr(tab, inv) > 30
    assert psnr(tab, inv) < 50