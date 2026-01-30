
import pywt

from .utils import affichage, noise
from .monteCarlo import monteCarlo, writeEstimated
from .curvelet import curveletTransfrom, inv_curveletTransform

def check_Bvalue(Bmin, J):
    list_B = []
    B = Bmin
    list_B.append(B)
    for j in range(J):
        if j % 2 == 0:
            B *= 2
            list_B.append(B)
    sigma = {}

    filename = "data/estimated.txt"

    with open(filename, "r") as f:
        for ligne in f:
            parts = ligne.strip().split()      # sépare par espace
            if parts:                          # vérifie que la ligne n'est pas vide
                sigma[int(parts[0])] = float(parts[1])

    print("Checking if all sigma has already been calculated")
    check = True
    
    for b in list_B:
        if b not in sigma:
            if check:
                print("All sigma has not been calculated this operation can be time consuming")
                check = False

            sigma[b] = monteCarlo(b, b, 10)
            writeEstimated(b, sigma[b])

    print("End of sigma check")

    return sigma
            

def filter(noisy_image, Bmin, J):
    sigma = check_Bvalue(Bmin, J)

    print("Applying curvelet transform")
    c, curvelet = curveletTransfrom(noisy_image, J, Bmin)


    print("Applying hard tresholding")
    
    for j in range(J):
        ridgeletJ = curvelet[j][0]

        B = 2**((j+1)//2)*Bmin
        k = 4 if j == 0 else 3
        value = sigma[B] * k * 4

        for b_id in range(len(ridgeletJ)):
            for r_id in range(len(ridgeletJ[b_id])):

                coeffs = ridgeletJ[b_id][r_id]
                
                thresholded_coeffs = [pywt.threshold(c, value, mode='hard', substitute = 0) for c in coeffs]

                ridgeletJ[b_id][r_id] = thresholded_coeffs
            
    print("applying inverse curvelet transform")

    return inv_curveletTransform(c, curvelet, J, Bmin, noisy_image.shape)
