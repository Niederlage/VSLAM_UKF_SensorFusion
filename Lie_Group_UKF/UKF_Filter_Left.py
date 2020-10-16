import numpy as np
from scipy.linalg import block_diag
from Lie_Group_UKF.LG_Tool import Lie_Group, cholupdate


def lukfPropagation(dt, chi, chiAnt, omega_b, a_b, S, omega, acc, Qc, g):
    # Left-UKF on Lie Groups
    q = len(S)  # state size
    S_aug = block_diag(S, Qc)
    N_aug = len(S_aug)

    # scaled unsented transform
    W0 = 1 - N_aug / 3
    Wj = (1 - W0) / (2 * N_aug)
    gamma = np.sqrt(N_aug / (1 - W0))

    # Prediction
    ichi = lg.invSE3(chi)
    omega = omega - omega_b  # unbiased inputs
    acc = acc - a_b
    X = gamma * np.hstack((np.zeros((N_aug, 1)), S_aug.T, -S_aug.T))  # sigma-points
    X[10 - 1:15, :] = X[10 - 1:15, :] + X[q + 7 - 1:q + 12, :] * dt  # add bias noise

    for j in range(2, 2 * N_aug + 1 + 1):
        xi_j = np.hstack((X[1 - 1:9, j - 1], X[16 - 1:q, j - 1]))  # do not take bias
        w_j = X[q + 1 - 1:N_aug, j - 1]
        omega_bj = X[10 - 1:12, j - 1]
        a_bj = X[13 - 1:15, j - 1]
        chi_j = np.dot(chiAnt, lg.expSE3(xi_j))  # nonlinear- Lie Group propagation
        Rot = np.dot(chi_j[1 - 1:3, 1 - 1:3], lg.expSO3((omega + w_j[1 - 1:3] - omega_bj) * dt))
        v = chi_j[1 - 1:3, 4] + (np.dot(Rot, (acc + w_j[4 - 1:6] - a_bj)) + g) * dt
        x = chi_j[1 - 1:3, 5] + v * dt
        chi = lg.state2chi(Rot, v, x, chi_j[1 - 1:3, 6 - 1:])
        Xi_j = np.dot(ichi, chi)  # can be more time efficient
        logXi_j = lg.logSE3(Xi_j)
        X[1 - 1:9, j - 1] = logXi_j[1 - 1:9, j - 1]  # propagated sigma points
        X[16 - 1:q, j - 1] = logXi_j[16 - 1:q, j - 1]

    X = np.sqrt(Wj) * X
    Rs = np.linalg.qr(X[1 - 1:q, 2 - 1:2 * N_aug + 1].T, mode='r')  # FIXME
    S = Rs[1 - 1:q, 1 - 1:q]

    return S


def hlukf(chi, xi, param, v):
    Pi = param.Pi
    chiC = param.chiC
    RotC = chiC[1 - 1:3, 1 - 1:3]
    xC = chiC[1 - 1:3, 4 - 1]

    yAmers = param.yAmers
    NbAmers = len(yAmers)

    chi_j = np.dot(chi, lg.expSE3(xi))
    Rot = chi_j[1 - 1:3, 1 - 1:3]
    x = chi_j[1 - 1:3, 5 - 1]
    PosAmers = chi_j[1 - 1:3, 6 - 1:]
    posAmers = PosAmers[:, yAmers]

    z = Pi * (np.dot((np.dot(Rot, RotC)).T, (posAmers - np.kron(x, np.ones((1, NbAmers))))) - np.kron(xC, np.ones(
        (1, NbAmers))))
    y = z[1 - 1:2, :] / z[3, :]
    y = y + v
    return y


def lukfUpdate(chi, omega_b, a_b, S, y, param, R, ParamFilter):
    param.Pi = ParamFilter.Pi
    param.chiC = ParamFilter.chiC

    k = len(y)
    q = len(S)
    N_aug = q + k
    Rc = np.linalg.cholesky(np.kron(np.eye(int(k / 2), R)))
    S_aug = block_diag(S, Rc)

    # scaled unsented transform
    W0 = 1 - N_aug / 3
    Wj = (1 - W0) / (2 * N_aug)
    gamma = np.sqrt(N_aug / (1 - W0))
    alpha = 1
    beta = 2

    # Compute transformed measurement
    X = gamma * np.hstack((np.zeros((N_aug, 1)), S_aug.T, -S_aug.T))  # sigma-points
    Y = np.zeros((k, 2 * N_aug + 1))
    Y[:, 1 - 1] = hlukf(chi, np.zeros(q - 6, 1), param, np.zeros((N_aug - q, 1)))

    for j in range(2, 2 * N_aug + 1 + 1):
        xi_j = np.hstack((X[1 - 1:9, j - 1], X[16 - 1:q, j - 1]))  # TODO if X[] as array[[]]
        v_j = X[q + 1 - 1:N_aug, j - 1]
        Y[:, j - 1] = hlukf(chi, xi_j, param, v_j)

    ybar = W0 * Y[:, 1 - 1] + Wj * np.sum(Y[:, 2 - 1:], 2)  # Measurement mean #TODO
    Y[:, 1 - 1] = np.sqrt(abs(W0 + (1 - alpha ** 2 + beta))) * (Y[:, 1 - 1] - ybar)
    YY = np.sqrt(Wj) * (Y[:, 2 - 1:2 * N_aug + 1] - np.dot(ybar, np.ones(1, 2 * N_aug)))
    Rs = np.linalg.qr(YY.T, mode='r')
    Ss = Rs[1 - 1:k, 1 - 1:k]
    Sy = cholupdate(Ss, Y[:, 1 - 1], '-')  # Sy'*Sy = Pyy #FIXME
    Pyy = np.dot(Sy.T, Sy)
    Pxy = np.zeros((q, k))
    for j in range(2, 2 * N_aug + 1 + 1):
        Pxy = Pxy + Wj * np.dot(X[1 - 1:q, j - 1], (Y[:, j - 1] - ybar).T)

    K = np.dot(Pxy, np.linalg.inv(Pyy))  # Gain

    xibar = np.dot(K, (y - ybar))
    omega_b = omega_b + xibar[10 - 1:12]
    a_b = a_b + xibar[13 - 1:15]
    xibar = xibar[1 - 1:9, 16 - 1:q]

    # Covariance update
    A = np.dot(K, Sy.T)
    for n in range(1, k + 1):  # TODO
        S = cholupdate(S, A[:, n - 1], '-')

    # Update mean state
    chi = np.dot(chi, lg.expSE3(xibar))
    J = np.dot(lg.xi2calJr(xibar))
    S = np.dot(S, J)

    return chi, omega_b, a_b, S


if __name__ == '__main__':
    lg = Lie_Group()
