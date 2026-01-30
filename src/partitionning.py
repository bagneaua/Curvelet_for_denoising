import sys
from skimage import io
import numpy as np

from .utils import affichage, noise

def partitioning(noisyImage, b):
    stride = b // 2
    blocks = []
    pos = []
    
    for i in range(0, noisyImage.shape[0]-b+1, stride):
        for j in range(0, noisyImage.shape[1]-b+1, stride):
            blocks.append(noisyImage[i:i+b, j:j+b])
            pos.append((i, j))
    
    return blocks, pos

def fenetre_cos2_1d(b):
    x = np.linspace(-1, 1, b)
    return np.cos(np.pi * x / 2) ** 2

def fenetre_cos2_2d(b):
    w1d = fenetre_cos2_1d(b)
    return np.outer(w1d, w1d)

def inv_partitioning(blocks, pos, image_shape, b):
    '''on multiplie la fenètre des cos par le block beaucoup plus rapide que de séquencer 
    pour chaque pixel (lenteur python vs array numpy c++) et même principe que l'article'''
    img = np.zeros(image_shape)
    poids = np.zeros(image_shape)

    W = fenetre_cos2_2d(b)

    for block, (i, j) in zip(blocks, pos):
        img[i:i+b, j:j+b] += W * block 
        poids[i:i+b, j:j+b] += W
    return img / poids



if __name__ == "__main__":
    nameFile = sys.argv[1]
    image = io.imread(nameFile)
    noisy_image = noise(image, 20)

    J = 5
    B = 16

    part, pos = partitioning(image, B)
    inv = inv_partitioning(part, pos, image.shape, B)

    affichage(image, noisy_image, inv)