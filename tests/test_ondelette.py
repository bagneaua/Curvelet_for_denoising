import numpy.random as rd

from src.utils import psnr
from src.ondelette import ondeletteTransform, inv_ondeletteTransform


def test_radon():
    tab = rd.rand(2*256,256)
    tab *= 255
    ond = ondeletteTransform(tab)
    inv = inv_ondeletteTransform(ond)
    assert psnr(tab, inv) > 300
    assert psnr(tab, inv) < 500