import matplotlib.pyplot as plt
import numpy as np

def affichage(image, noisyImage, filteredImage):
    psnrNoisy = psnr(image, noisyImage)
    psnrFiltered = psnr(image, filteredImage)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image without noise')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisyImage, cmap='gray')  
    plt.title(f'Image with noise\nPSNR = {psnrNoisy} dB')    
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(filteredImage, cmap='gray')
    plt.title(f'Filtered image\nPSNR = {psnrFiltered} dB')
    plt.axis('off')

    plt.show()


def noise(image, sigma):
    noisyImage = image + sigma * np.random.randn(*image.shape)
    return noisyImage

def psnr(image1, image2, pmax=255.0):
    sum = np.sum((image1 - image2)**2)
    if sum == 0:
        return np.inf
    sum /= (image1.shape[0] * image1.shape[1])
    return 10*np.log10(pmax**2/sum)

def affichage2(image1, image2, string=""):
    mse_images = mse(image1, image2)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')  
    plt.title(f'inv_{string}({string}(Image))\nMSE={mse_images:.5f}')    
    plt.axis('off')

    plt.show()

def mse(image1, image2):
    return np.linalg.norm(image1-image2, 2) / np.linalg.norm(image1, 2)
