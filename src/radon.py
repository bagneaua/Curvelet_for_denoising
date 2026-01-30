import sys
from skimage import io
import numpy as np

from .utils import affichage, noise

def cartesianToPolar(fft2):
    n = fft2.shape[0]
    cx, cy = n//2, n//2
    fft_polar = np.zeros((2*n, n), dtype=complex)

    for l_id in range(2*n):
        #On tire les points d'où parte les rayons

        if l_id < n:
            #bord haut à bord bas
            px, py = -n//2 + l_id, n//2
        else:
            #bord droit à bord gauche 
            px, py = n//2, n + n//2 - l_id

        for m in range(n):
            inter = m - n//2
            if py == n//2:
                # on tire du haut angle avec axe vertical <= pi/4
                x = inter * px/py if py != 0 else 0 #droite d'eq y = py/px * x mais on veut x en fonctino de y
                y = inter
            else:
                 #on tire du côté 
                x = inter
                y = inter * py / px if px != 0 else 0
            
            # coord dans la grille nearest neighboor pas d'interpolation linéaire dans l'article
            fx = int(np.round(x)) + cx
            fy = int(np.round(y)) + cy

            fx = np.clip(fx, 0, n-1)
            fy = np.clip(fy, 0, n-1)
            
            fft_polar[l_id, m] = fft2[fy, fx]
    return fft_polar
    
def radonTransform(block):
    n = block.shape[0]
    fft2 = np.fft.fftshift(np.fft.fft2(block)) #on commence par la fft2d

    fft_polar = cartesianToPolar(fft2)

    radon = np.zeros_like(fft_polar, dtype=float)
    for l_id in range(2 * n):
        radon[l_id, :] = np.real(np.fft.ifft(fft_polar[l_id, :]))

    return radon

def polarToCartesian(fft_polar):
    n = fft_polar.shape[1]
    cx, cy = n//2, n//2
    fft2 = np.zeros((n, n), dtype=complex) # stock res complex car fourier
    count = np.zeros((n, n))  # Nombre de contributions par pixel


    for l_id in range(2*n):
        #pareil que precedemment
        #On tire les points d'où parte les rayons

        if l_id < n:
            #bord haut à bord bas
            px, py = -n//2 + l_id, n//2
        else:
            #bord droit à bord gauche 
            px, py = n//2, n + n//2 - l_id

        for m in range(n):
            inter = m - n//2
            if py == n//2:
                # on tire du haut angle avec axe vertical <= pi/4
                x = inter * px/py if py != 0 else 0 #droite d'eq y = py/px * x mais on veut x en fonctino de y
                y = inter
            else:
                 #on tire du côté 
                x = inter
                y = inter * py / px if px != 0 else 0
            
            # coord dans la grille nearest neighboor pas d'interpolation linéaire dans l'article
            fx = int(np.round(x)) + cx
            fy = int(np.round(y)) + cy

            if 0 <= fx < n and 0 <= fy < n:
                fft2[fy, fx] += fft_polar[l_id, m]
                count[fy, fx] += 1

    mask = count > 0
    fft2[mask] /= count[mask]
        
    return fft2


def inv_radonTransfrom(radon):
    n = radon.shape[1]
    fft_polar = np.zeros((2 *n,n), dtype=complex)
    for theta_idx in range(2 *n):
        fft_polar[theta_idx, :] = np.fft.fft(radon[theta_idx, :])

    fft2 = polarToCartesian(fft_polar)

    return np.real(np.fft.ifft2(np.fft.ifftshift(fft2)))


if __name__ == "__main__":
    nameFile = sys.argv[1]
    image = io.imread(nameFile)
    noisy_image = noise(image, 20)

    J = 5
    B = 16

    radon = radonTransform(image)
    inv = inv_radonTransfrom(radon)

    affichage(image, noisy_image, inv)