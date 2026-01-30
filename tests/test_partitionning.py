import numpy.random as rd

from src.partitionning import partitioning, inv_partitioning
from src.utils import psnr

def test_atrous():
    tab = rd.rand(256,256)
    tab *= 255
    blocks, pos = partitioning(tab, 16)
    inv = inv_partitioning(blocks, pos, tab.shape, 16)
    assert psnr(tab, inv) > 200
    assert psnr(tab, inv) < 500