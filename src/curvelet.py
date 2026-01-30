import sys
from skimage import io

from .utils import affichage, noise
from .partitionning import partitioning, inv_partitioning
from .atrous import atrous_transform, inv_atrous
from .ridgelet import ridgeletTransform, inv_ridgeletTransform


def curveletTransform(noisyImage, J, Bmin):
    c , w = atrous_transform(noisyImage, J)
    B = Bmin
    curvelet = []

    for j in range(J):
        ridgelet_scaleJ = []
        pos_scaleJ = []

        blocks, pos = partitioning(w[j], B)
        ridgelet_blocks = []

        for block in blocks:
            ridgelet_on_block = ridgeletTransform(block)
            ridgelet_blocks.append(ridgelet_on_block)

        ridgelet_scaleJ.append(ridgelet_blocks)
        pos_scaleJ.append(pos)  # stocke les positions du bloc

        curvelet.append((ridgelet_scaleJ, pos_scaleJ))

        if j % 2 == 1:
            B *= 2

    return c, curvelet  # renvoyer aussi l'approximation grossière

def inv_curveletTransform(c, curvelet, J, Bmin, image_shape):
    B = Bmin

    for j in range(J):
        ridgelet_scaleJ, pos_scaleJ = curvelet[j]
        w_scaleJ = []

        for ridgelet_blocks, pos in zip(ridgelet_scaleJ, pos_scaleJ):
            # Inverse Ridgelet sur chaque bloc
            blocks_rec = [inv_ridgeletTransform(r) for r in ridgelet_blocks]

            # Recoller les blocs avec fenêtre cosinus
            wj = inv_partitioning(blocks_rec, pos, image_shape, B)
            w_scaleJ.append(wj)

        if j % 2 == 1:
            B *= 2

    # Inverse Atrous pour reconstruire l'image finale
    return inv_atrous(c, w_scaleJ)


if __name__ == "__main__":
    nameFile = sys.argv[1]
    image = io.imread(nameFile)
    noisy_image = noise(image, 20)

    J = 5
    B = 16

    c, curvelet = curveletTransform(image, J, B)
    inv = inv_curveletTransform(c, curvelet, J, B, image.shape)

    affichage(image, noisy_image, inv)