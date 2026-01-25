from src.utils import partitioning, inv_partitioning, psnr
import numpy.random as rd

def test_atrous():
    tab = rd.rand(256,256)
    blocks, pos = partitioning(tab, 16)
    inv = inv_partitioning(blocks, pos, tab.shape, 16)
    assert psnr(tab, inv) > 200
    assert psnr(tab, inv) < 500