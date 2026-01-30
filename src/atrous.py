import sys
from skimage import io
import numpy as np
from scipy.signal import convolve2d

from .utils import affichage, noise

def atrous_filter(h, j):
    if j == 0:
        return h.copy()
    
    step = 2**(j-1)
    hj = np.zeros((len(h) + (len(h)-1)*(step-1),))
    hj[::step] = h
    return hj


def convolve2d_separable(image, filt):
    temp = convolve2d(image, filt[:, None], mode='same', boundary='symm')
    result = convolve2d(temp, filt[None, :], mode='same', boundary='symm')
    return result

def atrous_transform(image, J):
    h = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0

    c = image.astype(np.float64)
    w = []

    for j in range(1, J+1):
        hj = atrous_filter(h, j)
        cj = convolve2d_separable(c, hj)
        wj = c - cj
        w.append(wj)
        c = cj

    return c, w

def inv_atrous(cJ, w):
    res = cJ.copy()
    for wj in w:
        res += wj
    return res

if __name__ == "__main__":
    nameFile = sys.argv[1]
    image = io.imread(nameFile)
    noisy_image = noise(image, 20)

    J = 5
    B = 16

    c, w = atrous_transform(image, J)
    inv = inv_atrous(c, w)

    affichage(image, noisy_image, inv)
