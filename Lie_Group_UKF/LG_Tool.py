import numpy as np
from scipy.special import bernoulli
from scipy.spatial.transform import Rotation as Rotm


def cholupdate(R, x, sign):
    p = np.size(x)
    x = x.reshape(-1)
    for k in range(p):
        if sign == '+':
            r = np.sqrt(R[k, k] ** 2 + x[k] ** 2)
        elif sign == '-':
            r = np.sqrt(R[k, k] ** 2 - x[k] ** 2)
        c = r / R[k, k]
        s = x[k] / R[k, k]
        R[k, k] = r
        if sign == '+':
            R[k, k + 1:p] = (R[k, k + 1:p] + s * x[k + 1:p]) / c
        elif sign == '-':
            R[k, k + 1:p] = (R[k, k + 1:p] - s * x[k + 1:p]) / c
        x[k + 1:p] = c * x[k + 1:p] - s * R[k, k + 1:p]
    return R


class Lie_Group:
    def __init__(self):
        self.tolerance = 1e-10

    def phi_clip(self, phi):
        phi %= 2 * np.pi
        while (abs(phi) > np.pi):
            phi -= 2 * np.pi

        return phi

    def chi2state(self, chi):
        Rot = chi[:3, :3]
        v = chi[:3, 3].reshape((3, 1))
        x = chi[:3, 4].reshape((3, 1))
        if np.size(chi, 1) > 5:
            PosAmers = chi[:3, 5:]  # TODO if Landmarks > 1 possible
        else:
            PosAmers = []

        return Rot, v, x, PosAmers

    def state2chi(self, Rot, v, x, PosAmers):
        if PosAmers is not None:
            NbAmers = np.size(PosAmers, 1)
            chi = np.eye(5 + NbAmers)
            chi[:3, :5] = np.hstack((Rot, v.reshape(3, 1), x.reshape(3, 1)))
            if NbAmers > 0:
                chi[:3, 5:] = PosAmers
            return chi
        else:
            chi = np.eye(5)
            chi[:3, :3] = Rot
            chi[:3, 3] = v
            chi[:3, 4] = x
            return chi

    def hat_operator(self, theta):
        vec_len = np.size(theta)
        if vec_len == 3:
            xi = np.array([[0, -theta[2], theta[1]],
                           [theta[2], 0, -theta[0]],
                           [-theta[1], theta[0], 0]])
        else:
            phi = theta[:3]
            rho = theta[3:]
            xi = np.zeros((vec_len // 3 + 2, vec_len // 3 + 2))
            xi[:3, :3] = np.array([[0, -phi[2], phi[1]],
                                   [phi[2], 0, -phi[0]],
                                   [-phi[1], phi[0], 0]])
            xi[:3, 3:] = rho.reshape(3, (vec_len // 3 - 1))
        return xi

    def curlyhat_operator(self, theta):
        vec_len = np.size(theta)
        phi = theta[:3]
        rho = theta[3:]
        xi = np.zeros((vec_len // 3 + 2, vec_len // 3 + 2))
        rot = self.hat_operator(phi)
        xi[:3, :3] = rot
        xi[:3, 3:] = self.hat_operator(rho)
        xi[3:, 3:] = rot
        return xi

    # with ph clip
    def expSO3(self, xi):
        theta = np.linalg.norm(xi)
        a_ = xi / theta
        theta = self.phi_clip(theta)
        if theta == 0:
            chi = np.eye(3)
        else:
            a_hat = self.hat_operator(a_)
            chi = np.eye(3) + np.sin(theta) * a_hat + (1 - np.cos(theta)) * a_hat @ a_hat

        return chi

    # with ph clip
    def expSE3(self, xi):
        ###########################################
        #     given phi must remain in (-pi,pi)
        ###########################################
        theta = np.linalg.norm(xi[:3])
        a_ = xi[:3] / theta
        theta = self.phi_clip(theta)
        NbXi = int(np.size(xi) / 3) - 1
        if NbXi < 2:
            if theta == 0:
                chi = np.eye(4)
                chi[:3, 3] = xi[3:]
            else:
                if theta < 0:
                    theta = -theta
                    # print("sign turns")
                Xi = np.zeros((4, 4))
                Xi[:3, :3] = self.hat_operator(theta * a_)
                Xi[:3, 3] = xi[3:]
                Xi2 = Xi @ Xi
                chi = np.eye(4) + Xi + (1 - np.cos(theta)) / theta ** 2 * Xi2 + (
                        theta - np.sin(theta)) / theta ** 3 * Xi2 @ Xi
        else:
            xi_col = xi[3:].reshape((NbXi, 3)).T
            if theta == 0:
                chi = np.eye(3 + NbXi)
                chi[:3, 3:] = xi_col
            else:
                if theta < 0:
                    theta = -theta
                    # print("sign turns")
                Xi = np.zeros((3 + NbXi, 3 + NbXi))
                Xi[:3, :3] = self.hat_operator(theta * a_)
                Xi[:3, 3:] = xi_col
                Xi2 = Xi @ Xi
                chi = np.eye(3 + NbXi) + Xi + (1 - np.cos(theta)) / theta ** 2 * Xi2 + (
                        theta - np.sin(theta)) / theta ** 3 * Xi2 @ Xi

        return chi

    def invSE3(self, chi):
        ichi = np.eye(len(chi))
        iR = chi[:3, :3].T
        ichi[:3, :3] = iR
        ichi[:3, 3:] = - iR @ chi[:3, 3:]
        return ichi

    # with ph clip
    def logSO3(self, R):
        theta = np.arccos((np.trace(R) - 1) / 2)
        theta = self.phi_clip(theta)
        if theta == 0:
            res = np.zeros((3,))
        else:
            t_hat = theta / (2 * np.sin(theta)) * (R - R.T)
            res = np.array([-t_hat[1, 2], t_hat[0, 2], -t_hat[0, 1]])
        return res

    def logSE3(self, chi):
        C = chi[:3, :3]
        r = chi[:3, 3:]
        eigs, vecs = np.linalg.eig(C)
        idx = np.argmin(abs(eigs - 1))  # the index von lambda_min -1
        a_ = vecs[:, idx]  # 3x1 vector
        phi = np.arccos(0.5 * (np.trace(C) - 1))
        if phi % (2 * np.pi) == 0:
            a_ = np.zeros((3,))
            rho = r.T.flatten()
            return np.hstack((phi * a_, rho))
        else:
            # quat = Rotm.from_matrix(C).as_quat()
            # q1 = Rotm.from_matrix(self.expSO3(phi * a_)).as_quat()
            # q2 = Rotm.from_matrix(self.expSO3(-phi * a_)).as_quat()
            # norm2 = np.linalg.norm(q2 - quat)
            # norm1 = np.linalg.norm(q1 - quat)
            norm2 = np.linalg.norm(self.expSO3(-phi * a_) - C)
            norm1 = np.linalg.norm(self.expSO3(phi * a_) - C)
            if norm2 < norm1:
                phi = -phi
                print("sign turns")
            # aaT = a_[:, None] @ a_[:, None].T
            # aup = self.hat_operator(a_)
            # iJ = phi / (2 * np.tan(phi / 2)) * np.eye(3) + (1 - phi / (2 * np.tan(phi / 2))) * aaT - phi / 2 * aup
            iJ = self.vec2jaclInv(phi * a_)
            rho = (iJ @ r).T.flatten()
            X = np.hstack((phi * a_, rho))
            return np.real(X)

    # with ph clip
    def vec2jacl(self, vec):

        if np.size(vec) == 3:
            ph = np.linalg.norm(vec)
            a_ = vec / ph
            ph = self.phi_clip(ph)
            if ph < self.tolerance:
                # If the angle is small, fall back on the series representation
                Jl = self.vec2jaclSeries(ph * a_, 10)
            else:
                aaT = a_[:, None] @ a_[:, None].T
                aup = self.hat_operator(a_)
                cph = (1 - np.cos(ph)) / ph  # without +-
                sph = np.sin(ph) / ph  # with +-
                Jl = sph * np.eye(3) + (1 - sph) * aaT + cph * aup  # sign attention
            return Jl

        elif np.size(vec) == 6:
            phi = vec[:3]
            ph = np.linalg.norm(phi)
            a_ = phi / ph
            ph = self.phi_clip(ph)
            if ph < self.tolerance:
                # If the angle is small, fall back on the series representation
                Jl = self.vec2jaclSeries(ph * a_, 10)
            else:
                Jsmall = self.vec2jacl(phi)
                Q = self.vec2Ql(vec)
                Jup = np.hstack((Jsmall, Q))
                Jdown = np.hstack((np.zeros((3, 3)), Jsmall))
                Jl = np.vstack((Jup, Jdown))
            return Jl

    def vec2jaclSeries(self, vec, N):
        if np.size(vec) == 3:
            Jl = np.eye(3)
            pxn = np.eye(3)
            px = self.hat_operator(vec)
            for n in range(1, N + 1):
                pxn = np.dot(pxn, px) / (n + 1)
                Jl = Jl + pxn
            return Jl

        elif np.size(vec) == 6:
            Jl = np.eye(6)
            pxn = np.eye(6)
            px = self.curlyhat_operator(vec)
            for n in range(1, N + 1):
                pxn = np.dot(pxn, px) / (n + 1)
                Jl = Jl + pxn
            return Jl

    # with ph clip
    def vec2jaclInv(self, vec):
        if np.size(vec) == 3:
            ph = np.linalg.norm(vec)
            a_ = vec / ph
            ph = self.phi_clip(ph)
            if ph < self.tolerance:
                # If the angle is small, fall back on the series representation
                invJl = self.vec2jaclInvSeries(ph * a_, 10)
            else:
                aaT = a_[:, None] @ a_[:, None].T
                aup = self.hat_operator(a_)
                ph_2 = 0.5 * ph
                phicot = ph_2 / np.tan(ph_2)  # without +-
                invJl = phicot * np.eye(3) + (1 - phicot) * aaT - ph_2 * aup  # sign attention

            return invJl

        elif np.size(vec) == 6:
            phi = vec[:3]
            ph = np.linalg.norm(phi)
            a_ = phi / ph
            ph = self.phi_clip(ph)
            if ph < self.tolerance:
                # If the angle is small, fall back on the series representation
                invJl = self.vec2jaclInvSeries(ph * a_, 10)
            else:
                invJsmall = self.vec2jaclInv(ph * a_)
                Q = self.vec2Ql(vec)
                invJQinvJ = invJsmall @ Q @ invJsmall
                invJup = np.hstack((invJsmall, - invJQinvJ))
                invJdown = np.hstack((np.zeros((3, 3)), invJsmall))
                invJl = np.vstack((invJup, invJdown))
            return invJl

    def vec2jaclInvSeries(self, vec, N):
        if np.size(vec) == 3:
            invJl = np.eye(3)
            pxn = np.eye(3)
            px = self.hat_operator(vec)
            for n in range(1, N + 1):
                pxn = np.dot(pxn, px) / n
                invJl = invJl + bernoulli(n)[n] * pxn

            return invJl
        elif np.size(vec) == 6:
            invJl = np.eye(6)
            pxn = np.eye(6)
            px = self.curlyhat_operator(vec)
            for n in range(1, N + 1):
                pxn = np.dot(pxn, px) / n
                invJl = invJl + bernoulli(n)[n] * pxn

            return invJl

    # with ph clip
    def vec2Ql(self, vec):
        rho = vec[:3]
        phi = vec[3:6]

        ph = np.linalg.norm(phi)
        a_ = phi / ph
        ph = self.phi_clip(ph)

        ph2 = ph * ph
        ph3 = ph2 * ph
        ph4 = ph3 * ph
        ph5 = ph4 * ph

        cph = np.cos(ph)
        sph = np.sin(ph)

        rx = self.hat_operator(rho)
        px = self.hat_operator(ph * a_)
        pxrx = px @ rx
        rxpx = rx @ px
        pxpx = px @ px

        m2 = (ph - sph) / ph3
        m3 = (0.5 * ph2 + cph - 1) / ph4
        m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5  # 0.5 * (m3 - 3 * (ph - sph - ph3 / 6) / ph5)

        t1 = 0.5 * rx
        t2 = m2 * (pxrx + rxpx + pxrx @ px)
        t3 = m3 * (px @ pxrx + rxpx @ px - 3 * pxrx @ px)
        t4 = m4 * (pxrx @ pxpx + pxpx @ rxpx)

        Q = t1 + t2 + t3 + t4

        return Q

    def vec2adjJl(self, xi):
        # Compute Left Jacobian
        Nxi = len(xi)
        NbAmers = int(Nxi / 3) - 3
        phi = xi[:3]  # TODO if order phi != rho
        calJl = np.zeros((3, Nxi))
        ph = np.linalg.norm(phi)
        a_ = phi / ph
        ph = self.phi_clip(ph)
        if ph < self.tolerance:
            # If the angle is small, fall back on the series representation
            Jl = self.vec2jaclSeries(phi, 10)
        else:
            aaT = a_[:, None] @ a_[:, None].T
            aup = self.hat_operator(a_)
            cph = (1 - np.cos(ph)) / ph
            sph = np.sin(ph) / ph
            Jl = sph * np.eye(3) + (1 - sph) * aaT + cph * aup

        calJl = np.kron(np.eye(NbAmers + 5), Jl)

        for i in range(2):
            Q = self.vec2Ql(np.hstack((ph * a_, xi[1 + 3 * i - 1:3 + 3 * i])))
            calJl[:3, 3 * (i + 1):3 * (i + 2)] = Q

        for i in range(NbAmers):
            Q = self.vec2Ql(np.hstack((ph * a_, xi[9 + 3 * i: 12 + 3 * i])))
            calJl[: 3, 15 + 3 * i: 18 + 3 * i] = Q

        calJl[9: 15, 9: 15] = np.eye(6)

        return calJl


# class Error():
#     def __init__(self, R, v, x, omega_b, a_b, rms):
#         self.R = R
#         self.v = v
#         self.x = x
#         self.omega_b = omega_b
#         self.a_b = a_b
#         self.rms = rms
#
#     def traj2error(self, trajFilter, trajReal):
#         NbMax = len(trajReal.psi)
#         self.R = np.zeros((3, NbMax))
#         rot = Rotation_local()
#         for i in range(NbMax):
#             eulerFilter = np.array([trajFilter.psi[i + 2], trajFilter.theta[i + 2], trajFilter.phi[i + 2]])
#             eulerReal = np.array([trajReal.psi[i + 2], trajReal.theta[i + 2], trajReal.phi[i + 2]])
#             RotFilter = rot.eul2Rotm(eulerFilter)
#             RotReal = Rotation.from_rotvec(eulerReal)
#             self.R[:, i] = logSO3(np.dot(RotFilter.T, RotReal))  # 'R3'
#
#         self.v = trajFilter.v[:, :NbMax] - trajReal.v[:, :NbMax]
#         self.x = trajFilter.x[:, :NbMax] - trajReal.x[:, :NbMax]
#         self.omega_b = trajFilter.omega_b[:, :NbMax] - trajReal.omega_b[:, :NbMax]
#         self.a_b = trajFilter.a_b[:, :NbMax] - trajReal.a_b[:, :NbMax]
#
#         # rms
#         R = np.zeros((NbMax, 1))
#         v = np.zeros((NbMax, 1))
#         x = np.zeros((NbMax, 1))
#         omega_b = np.zeros((NbMax, 1))
#         a_b = np.zeros((NbMax, 1))
#
#         for i in range(NbMax):
#             R[i] = np.linalg.norm(self.R[:, i]) / 3 * 180 / np.pi
#             v[i] = np.linalg.norm(self.v[:, i]) / 3
#             x[i] = np.linalg.norm(self.x[:, i]) / 3
#             omega_b[i] = np.linalg.norm(self.omega_b[:, i]) / 3
#             a_b[i] = np.linalg.norm(self.a_b[:, i]) / 3
#
#         self.rms.R = R
#         self.rms.v = v
#         self.rms.x = x
#         self.rms.omega_b = omega_b
#         self.rms.a_b = a_b
if __name__ == '__main__':
    test_expso3 = True
    test_expse3 = False
    test_jacobi = False
    lg = Lie_Group()
    alpha = np.array([-np.sqrt(0.4) * np.pi / 5, np.sqrt(0.1) * np.pi / 5, np.sqrt(0.5) * np.pi / 5])
    beta = np.array([0.5 * np.pi, 0, 0])
    xp = np.array([-np.sqrt(0.4) * np.pi / 5, np.sqrt(0.1) * np.pi / 5, np.sqrt(0.5) * np.pi / 5, -11.2, 46, 10])

    if test_expso3:
        C1 = lg.expSO3(alpha)
        C2 = lg.expSO3(beta)
        phi12 = np.linalg.norm(lg.logSO3(C1.T @ C2))
        print("delta phi: = ", phi12 / np.pi)
        print("R=\n", lg.expSO3(xp[:3]))
        print("vec\n", lg.logSO3(lg.expSO3(xp[:3])))

    if test_expse3:
        T_a = lg.expSE3(xp)
        test_a = lg.logSE3(T_a)
        print("Ta=\n", T_a)
        print("test_a=\n", test_a)

    if test_jacobi:
        Jotseries = lg.vec2jaclSeries(alpha, 10)
        InvJotseries = lg.vec2jaclInvSeries(alpha, 10)
        Jot = lg.vec2jacl(alpha)
        InvJot = lg.vec2jaclInv(alpha)
        print("jacobi series:\n", Jotseries @ InvJotseries)
        print("jacobi :\n", Jot @ InvJot)

# def vec2jacr(vec):
#     self.tolerance = 1e-12
#     if np.size(vec, ) == 3:
#         phi = vec
#         ph = np.linalg.norm(phi)
#         if ph < self.tolerance:
#             # If the angle is small, fall back on the series representation
#             Jr = vec2jacrSeries(phi, 10)
#         else:
#             axis = phi / ph  # with +-
#             cph = (1 - np.cos(ph)) / ph  # without +-
#             sph = np.sin(ph) / ph  # without +-
#             Jr = sph * np.eye(3) + (1 - sph) * np.dot(axis, axis.T) - cph * hat(axis)
#
#     elif np.size(vec, ) == 6:
#         # rho = vec[:3]
#         phi = vec[3:6]
#         ph = np.linalg.norm(phi)
#         if ph < self.tolerance:
#             # If the angle is small, fall back on the series representation
#             Jr = vec2jacrSeries(phi, 10)
#         else:
#             Jsmall = vec2jacr(phi)
#             Q = vec2Qr(vec)
#             Jup = np.hstack((Jsmall, Q))
#             Jdown = np.hstack((np.zeros((3, 3)), Jsmall))
#             Jr = np.vstack((Jup, Jdown))
#
#     return Jr
#
#
# def vec2jacrInv(vec):
#     self.tolerance = 1e-12
#     if np.size(vec, ) == 3:
#         phi = vec
#         ph = np.linalg.norm(phi)
#         if ph < self.tolerance:
#             # If the angle is small, fall back on the series representation
#             invJr = vec2jacrInvSeries(phi, 10)
#         else:
#             axis = phi / np.linalg.norm(phi)
#             ph_2 = 0.5 * ph
#             phicot = ph_2 * (1 / np.tan(ph_2))
#             invJr = phicot * np.eye(3) + (1 - phicot) * np.dot(axis, axis.T) + ph_2 * hat(axis)
#
#     elif np.size(vec, ) == 6:
#         # rho = vec[:3]
#         phi = vec[3:6]
#
#         ph = np.linalg.norm(phi)
#         if ph < self.tolerance:
#             # If the angle is small, fall back on the series representation
#             invJr = vec2jacrInvSeries(phi, 10)
#         else:
#             invJsmall = vec2jacrInv(phi)
#             Q = vec2Qr(vec)
#             invJsmallQ = np.dot(invJsmall, Q)
#             invJr = np.vstack((np.hstack((invJsmall, - np.dot(invJsmallQ, invJsmall))), \
#                                np.hstack((np.zeros((3, 3)), invJsmall))))
#
#     return invJr
#
# def vec2jacrSeries(vec, N):
#     if np.size(vec, ) == 3:
#         Jr = np.eye(3)
#         pxn = np.eye(3)
#         px = -hat(vec)
#         for n in range(1, N + 1):
#             pxn = np.dot(pxn, px) / (n + 1)
#             Jr = Jr + pxn
#
#     elif np.size(vec, ) == 6:
#         Jr = np.eye(6)
#         pxn = np.eye(6)
#         px = -curlyhat(vec)
#         for n in range(1, N + 1):
#             pxn = np.dot(pxn, px) / (n + 1)
#             Jr = Jr + pxn
#
#     return Jr
#
#
# def vec2jacrInvSeries(vec, N):
#     if np.size(vec, ) == 3:
#         invJr = np.eye(3)
#         pxn = np.eye(3)
#         px = -hat(vec)
#         for n in range(1, N + 1):
#             pxn = np.dot(pxn, px) / n
#             invJr = invJr + bernoulli(n)[n] * pxn
#
#     elif np.size(vec, ) == 6:
#         invJr = np.eye(6)
#         pxn = np.eye(6)
#         px = -curlyhat(vec)
#         for n in range(1, N + 1):
#             pxn = np.dot(pxn, px) / n
#             invJr = invJr + bernoulli(n)[n] * pxn
#
#     return invJr
# def vec2Qr(vec):
#     rho = -vec[:3]
#     phi = -vec[3:6]
#
#     ph = np.linalg.norm(phi)
#     ph2 = ph * ph
#     ph3 = ph2 * ph
#     ph4 = ph3 * ph
#     ph5 = ph4 * ph
#
#     cph = np.cos(ph)
#     sph = np.sin(ph)
#
#     rx = hat(rho)
#     px = hat(phi)
#     pxrx = np.dot(px, rx)
#     rxpx = np.dot(rx, px)
#     pxpx = np.dot(px, px)
#
#     m2 = (ph - sph) / ph3
#     m3 = (0.5 * ph2 + cph - 1) / ph4
#     m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5  # 0.5 * (m3 - 3 * (ph - sph - ph3 / 6) / ph5)
#
#     t1 = 0.5 * rx
#     t2 = m2 * (pxrx + rxpx + np.dot(pxrx, px))
#     t3 = m3 * (np.dot(px, pxrx) + np.dot(rxpx, px) - 3 * np.dot(pxrx, px))
#     t4 = m4 * (np.dot(pxrx, pxpx) + np.dot(pxpx, rxpx))
#
#     Q = t1 + t2 + t3 + t4
#
#     return Q
#
#
#
#
#
# def xi2calJr(xi):
#     # Compute Right Jacobian
#     taille_xi = len(xi)
#     self.tolerance = 1e-12
#     NbAmers = int(taille_xi / 3) - 3
#     phi = xi[:3]  # TODO if order phi != rho
#     calJr = np.zeros((3, taille_xi))
#     ph = np.linalg.norm(phi)
#     if ph < self.tolerance:
#         # If the angle is small,  fall back on the series representation
#         Jr = vec2jacrSeries(phi, 10)
#     else:
#         axis = phi / ph
#
#         cph = (1 - np.cos(ph)) / ph
#         sph = np.sin(ph) / ph
#
#         Jr = sph * np.eye(3) + (1 - sph) * np.dot(axis, axis.T) - cph * hat(axis)
#         for i in range(1, 2 + 1):
#             Q = vec2Qr(np.hstack((phi, xi[1 + 3 * i - 1:3 + 3 * i])))
#             calJr[:3, 1 + 3 * i - 1:3 + 3 * i] = Q
#
#         for i in range(1, NbAmers + 1):
#             Q = vec2Qr(np.hstack((phi, xi[7 + 3 * i - 1: 9 + 3 * i])))
#             calJr[: 3, 13 + 3 * i - 1: 15 + 3 * i] = Q
#
#     calJr = np.kron(np.eye(NbAmers + 5), Jr)
#     calJr[10 - 1: 15, 10 - 1: 15] = np.eye(6)
#
#     return calJr
#
#
#
# class Rotation_local:
#     def __init__(self):
#         pass
#
#     def eul2Rotm(self, eulerAngle):
#         eulerAngle = eulerAngle.reshape(3, 1)
#         alpha = eulerAngle[0, 0]
#         beta = eulerAngle[1, 0]
#         gamma = eulerAngle[2, 0]
#         Rx = np.array([[1, 0, 0],
#                        [0, np.cos(alpha), -np.sin(alpha)],
#                        [0, np.sin(alpha), np.cos(alpha)]])
#         Ry = np.array([[np.cos(beta), 0, -np.sin(beta)],
#                        [0, 1, 0],
#                        [-np.sin(beta), 0, np.cos(beta)]])
#         Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
#                        [np.sin(gamma), np.cos(gamma), 0],
#                        [0, 0, 1]])
#         R = np.dot(np.dot(Rz, Ry), Rx)
#         return R
#
#     def Rotm2eul(self, Rot):
#         Sy = np.sqrt(Rot[2, 1] ** 2 + Rot[2, 2] ** 2)
#         alpha = np.arctan2(Rot[2, 1], Rot[2, 2])
#         beta = np.arctan2(-Rot[2,], Sy)
#         gamma = np.arctan2(Rot[1,], Rot[0,])
#         phi = np.array([alpha, beta, gamma])
#         phi = phi.reshape(3, 1)
#         return phi
#
