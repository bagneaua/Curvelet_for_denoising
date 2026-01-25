from src.utils import ridgeletTransform, inv_ridgeletTransform, psnr
import numpy.random as rd

def test_ridgelet():
    tab = rd.rand(256,256)
    tab *= 255
    ridgelet = ridgeletTransform(tab)
    inv = inv_ridgeletTransform(ridgelet)
    assert psnr(tab, inv) > 50
    assert psnr(tab, inv) < 100