import sys
from skimage import io
import numpy as np

from .utils import affichage, noise
from .partitionning import partitioning, inv_partitioning
from .atrous import atrous_transform, inv_atrous
from .ridgelet import ridgeletTransform, inv_ridgeletTransform

def curveletTransfrom(noisyImage, J, Bmin):
    B = Bmin

    c , w = atrous_transform(noisyImage, J)
    curvelet = []

    for j in range(J):
        ridgeletJ = []
        blockScaleJ , posJ = partitioning(w[j], B)

        for block in blockScaleJ:
            ridgeletJ.append(ridgeletTransform(block))

        curvelet.append((ridgeletJ, posJ))

        if j % 2 == 1:
            B *= 2

    return c, curvelet

def inv_curveletTransform(c, curvelet, J, Bmin, image_shape):
    B = Bmin
    w = []

    for j in range(J):
        ridgeletJ = curvelet[j][0] #par soucis de claireté du code 
        posJ = curvelet[j][1]
        blockJ = []

        for ridgelet in ridgeletJ:
            blockJ.append(inv_ridgeletTransform(ridgelet))
        
        w.append(inv_partitioning(blockJ, posJ, image_shape, B))

        if j % 2 == 1:
            B *= 2
    
    return inv_atrous(c, w)

if __name__ == "__main__":
    nameFile = sys.argv[1]
    image = io.imread(nameFile)
    noisy_image = noise(image, 20)

    J = 5
    B = 16

    c, curvelet = curveletTransfrom(image, J, B)
    inv = inv_curveletTransform(c, curvelet, J, B, image.shape)

    affichage(image, noisy_image, inv)