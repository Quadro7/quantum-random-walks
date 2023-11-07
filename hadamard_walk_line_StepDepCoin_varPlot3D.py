import numpy as np
import math, cmath
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def quantum_walk(T, phi, beta):
    # SETUP
    q = 1/math.sqrt(2)
    H = np.matrix([[q, q], [q, -q]])
    # Wavefunction coefficients
    a = q
    b = q
    psi = np.matrix('0 ' * T + str(a) + ' 0' * T + ' ' + '0 ' * T + str(b*cmath.exp(1j*phi*math.pi/180)) + ' 0' * T).T # |psi> = a*|up> otimes |x=0> + b*exp(phi*pi/180)*|down> otimes |x=0>

    # GENERATING MATRICES
    idMatrix = np.identity(2*T+1)
    sD = np.zeros((2*T+1, 2*T+1)) # Sd
    sU = np.zeros((2*T+1, 2*T+1)) # Su
    for i in range(0, 2*T):
        sD.itemset((i, i+1), 1)
        sU.itemset((i + 1, i), 1)
    S = np.kron(np.matrix([[1, 0], [0, 0]]), sU) + np.kron(np.matrix([[0, 0], [0, 1]]), sD)
    # position dependent coin exp(1j*x*beta)
    coin = np.zeros((4*T+2, 4*T+2), dtype='complex')
    for i in range(2*T+1): # 0 to 2*T
        coin.itemset((i, i), q*cmath.exp(1j*beta*(i - T)))
        coin.itemset((i, 2*T + 1 + i), q*cmath.exp(1j*beta*(i - T)))
        coin.itemset((2*T + 1 + i, i), q*cmath.exp(1j*beta*(i - T)))
    for i in range(2*T+1):
        coin.itemset((i + 2*T + 1, i + 2*T + 1), -q*cmath.exp(1j*beta*(i - T)))
    U = np.dot(S, coin)
    # Position operator
    x = np.zeros((2*T+1, 2*T+1))
    for i in range(2*T+1):
        x.itemset((i, i), i - T)
    xC = np.kron(np.identity(2), x) # prepared matrix with identity in the coin space
    uT = U
    for _ in range(T-1):
        uT = np.dot(uT, U)
    # Running evolution
    psiT = np.dot(uT, psi)
    print("Psi inner product:", np.dot(psiT.H, psiT).item()) # Check if psiT is normalized
    varFirstTerm = np.dot(np.dot(psiT.H, np.dot(xC, xC)), psiT).item()
    varSecondTerm = (np.dot(np.dot(psiT.H, xC), psiT).item()) * np.conj(np.dot(np.dot(psiT.H, xC), psiT).item())
    variance = varFirstTerm - varSecondTerm
    print(f"Variance @T={T}", np.real(variance))
    probCoinUp = []
    probCoinDown = []
    for i in range(0, 2*T+1):
        probCoinUp.append(psiT[i].item() * np.conj(psiT[i].item()))
    for i in range(2*T+1, 4*T+2):
        probCoinDown.append(psiT[i].item() * np.conj(psiT[i].item()))

    # Prepping the x and y axis
    xpoints = []
    result = []
    for i in range(-T, T + 1):
        xpoints.append(i)
    for i in range(0, 2*T+1):
        result.append(probCoinUp[i] + probCoinDown[i])
    return [xpoints, result, variance]

if __name__ == "__main__":
    steps = 10
    xpoints = []
    ypoints = []
    zpoints = []
    for i in range(181):
        for j in range(91):
            distribution = quantum_walk(T=steps, phi=j, beta=math.pi / 180 *i)
            xpoints.append(i)
            ypoints.append(j)
            zpoints.append(np.real(distribution[2]))
    # PLOTTING
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    #ax.scatter(xpoints, ypoints, zpoints, alpha=0.2)
    ax.plot_trisurf(xpoints, ypoints, zpoints)
    plt.show()