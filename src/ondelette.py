import sys
from skimage import io
import numpy as np
import pywt

from .utils import affichage, noise


def ondeletteTransform(radon_block, wavelet='db1', level=None):
    n_angles = radon_block.shape[0]
    ridgelet_coeffs = []

    for i in range(n_angles):
        line = radon_block[i, :]
        coeffs = pywt.wavedec(line, wavelet=wavelet, level=level)
        ridgelet_coeffs.append(coeffs)

    return ridgelet_coeffs

def inv_ondeletteTransform(ridgelet_coeffs, wavelet='db1'):
    radon_block = []

    for coeffs in ridgelet_coeffs:
        line_rec = pywt.waverec(coeffs, wavelet=wavelet)
        radon_block.append(line_rec) 

    return np.array(radon_block)

if __name__ == "__main__":
    nameFile = sys.argv[1]
    image = io.imread(nameFile)
    noisy_image = noise(image, 20)

    J = 5
    B = 16

    ond = ondeletteTransform(image)
    inv = inv_ondeletteTransform(ond)

    affichage(image, noisy_image, inv)