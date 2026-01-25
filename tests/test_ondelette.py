from src.utils import ondelette, inv_ondelette, psnr
import numpy.random as rd

def test_radon():
    tab = rd.rand(2*256,256)
    ond = ondelette(tab)
    inv = inv_ondelette(ond)
    assert psnr(tab, inv) > 300
    assert psnr(tab, inv) < 500