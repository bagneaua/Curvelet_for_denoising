import sys
from skimage import io

from .utils import affichage2
from .radon import radonTransform, inv_radonTransfrom
from .ondelette import ondeletteTransform, inv_ondeletteTransform

def ridgeletTransform(block):
    radon = radonTransform(block)
    return ondeletteTransform(radon)

def inv_ridgeletTransform(ridgelet):
    radon = inv_ondeletteTransform(ridgelet)
    return inv_radonTransfrom(radon)

if __name__ == "__main__":
    nameFile = sys.argv[1]
    image = io.imread(nameFile)

    J = 5
    B = 16

    ond = ridgeletTransform(image)
    inv = inv_ridgeletTransform(ond)

    affichage2(image, inv)