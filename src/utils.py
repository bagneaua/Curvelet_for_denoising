import sys
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import convolve2d

def affichage(image, noisyImage, filteredImage):
    # psnrNoisy = psnr(image, noisyImage)
    psnrFiltered = psnr(image, filteredImage)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image without noise')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisyImage, cmap='gray')  
    # plt.title(f'Image with noise\nPSNR = {psnrNoisy} dB')    
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(filteredImage, cmap='gray')
    plt.title(f'Filtered image\nPSNR = {psnrFiltered} dB')
    plt.axis('off')

    plt.show()

def noise(image, sigma):
    noisyImage = image + sigma * np.random.randn(*image.shape)
    return noisyImage

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
    pour chaque pixel et même principe que l'article'''
    img = np.zeros(image_shape)
    poids = np.zeros(image_shape)

    W = fenetre_cos2_2d(b)

    for block, (i, j) in zip(blocks, pos):
        img[i:i+b, j:j+b] += W * block 
        poids[i:i+b, j:j+b] += W
    return img / poids

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

def ondelette(radon_block, wavelet='db1', level=None):
    n_angles = radon_block.shape[0]
    ridgelet_coeffs = []

    for i in range(n_angles):
        line = radon_block[i, :]
        coeffs = pywt.wavedec(line, wavelet=wavelet, level=level)
        ridgelet_coeffs.append(coeffs)

    return ridgelet_coeffs

def inv_ondelette(ridgelet_coeffs, wavelet='db1'):
    radon_block = []

    for coeffs in ridgelet_coeffs:
        line_rec = pywt.waverec(coeffs, wavelet=wavelet)
        radon_block.append(line_rec) 

    return np.array(radon_block)

def ridgeletTransform(block):
    radon = radonTransform(block)
    return ondelette(radon)

def inv_ridgeletTransform(ridgelet):
    radon = inv_ondelette(ridgelet)
    return inv_radonTransfrom(radon)

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


def psnr(image1, image2, pmax=255.0):
    sum = np.sum((image1 - image2)**2)
    if sum == 0:
        return np.inf
    sum /= (image1.shape[0] * image1.shape[1])
    return 10*np.log10(pmax**2/sum)

if __name__ == "__main__":
    nameFile = sys.argv[1]
    image = io.imread(nameFile)
    noisy_image = noise(image, 20)

    J = 5
    B = 16
    # c, curvelet = curveletTransform(image, J, B)
    # inv = inv_curveletTransform(c, curvelet, J, B, image.shape)

    # radon = radonTransform(image)
    # inv = inv_radonTransfrom(radon)

    # polar = cartesianToPolar(image)
    # inv = polarToCartesian(polar)

    # ridgelet = ridgeletTransform(image)
    # inv = inv_ridgeletTransform(ridgelet)

    # ond = ondelette(image)
    # inv = inv_ondelette(ond)

    # c, w = atrous_transform(image, J)
    # inv = inv_atrous(c, w)

    # part, pos = partitioning(image, B)
    # inv = inv_partitioning(part, pos, image.shape, B)



    # affichage(image, noisy_image, inv)

