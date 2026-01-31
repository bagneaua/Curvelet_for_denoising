import sys
from skimage import io
from .utils import noise, affichage
from .filter import filter


if __name__ == "__main__":
    nameFile = sys.argv[1]
    image = io.imread(nameFile)
    noisy_image = noise(image, 20)

    J = 5
    B = 16

    filtered = filter(noisy_image, B, J)
    affichage(image, noisy_image, filtered)