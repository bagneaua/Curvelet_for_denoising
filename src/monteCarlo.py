import sys
import numpy as np
import numpy.random as rd

from .ridgelet import ridgeletTransform


def generate_noisy(N, M, K):
    noisy_ridgelet = []
    for _ in range(K):
        noise_img = rd.random((N,M))
        noisy_ridgelet.append(ridgeletTransform(noise_img))
    return noisy_ridgelet

def monteCarlo(N, M, K):
    noisy_ridgelet = generate_noisy(N, M, K)
    means = []

    for rid in noisy_ridgelet:
        rid_means = [np.mean(np.concatenate(subBlock)**2) for subBlock in rid]
        means.append(np.mean(rid_means))
    
    return np.sqrt(np.mean(means))

def monteCarloAngle(N, M, K):
    noisy_ridgelet = generate_noisy(N, M, K)
    liste_sigma = np.zeros(2*N)

    for rid in noisy_ridgelet:
        for l_id in range(len(rid)): #on parcourt les angles
            print(np.concatenate(rid[l_id])**2)
            liste_sigma[l_id] += np.mean(np.concatenate(rid[l_id])**2)
    
    return np.sqrt(liste_sigma/K)


    

def writeEstimated(N, estimate):
    with open('data/estimated.txt', 'a') as f:
        f.write(str(N) + ' ' + str(estimate) + '\n')



if __name__ == "__main__":
    N = int(sys.argv[1])
    M = int(sys.argv[2])
    K = int(sys.argv[3])

    # estimate = monteCarlo(N, N, K)

    # writeEstimated(N, estimate)
    # print(estimate)

    estimate = monteCarloAngle(N, N, K)
    print(estimate)
