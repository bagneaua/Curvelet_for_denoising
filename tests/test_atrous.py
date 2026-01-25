from src.utils import atrous_transform, inv_atrous, psnr
import numpy.random as rd


def test_atrous():
    tab = rd.rand(256,256)
    tab *= 255
    c, w = atrous_transform(tab, 8)
    inv = inv_atrous(c, w)
    assert psnr(tab, inv) > 200
    assert psnr(tab, inv) < 500