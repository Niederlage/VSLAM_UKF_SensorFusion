import numpy as np
from scipy.linalg import block_diag
import scipy
from Lie_Group_UKF.LG_Tool import Lie_Group, cholupdate
import mathutils
from scipy.spatial.transform import Rotation


class Hybrid_KF_Right_Filter():

    def __init__(self, xi0, bias0, timestamp, iter_steps, ERROR_CHECK):
        self.g = np.array([0, 0, -9.80665])
        self.xi0 = xi0
        self.bias0 = bias0
        self.lg = Lie_Group()
        self.ERROR_CHECK = ERROR_CHECK
        self.iter_steps = iter_steps

        # init covariance
        p0Rot = (0.01 * np.pi / 180) ** 2
        p0v = 1.e-4
        p0x = 1.e-8
        p0omegab = 1.e-6
        p0ab = 1.e-6
        P0 = np.concatenate((p0Rot * np.ones((3,)), p0v * np.ones((3,)), p0x * np.ones((3,)), p0omegab * np.ones((3,)),
                             p0ab * np.ones((3,))))
        self.P0 = np.diag(P0)
        self.S0 = np.linalg.cholesky(self.P0)

        # init pocess noise
        q_omega = (1.6968e-4) ** 2 * 200
        q_a = (2e-3) ** 2 * 200
        q_omegab = (1.9393e-5) ** 2 * 200
        q_ab = (3e-3) ** 2 * 200
        Q0 = np.concatenate(
            (q_omega * np.ones((3,)), q_a * np.ones((3,)), q_omegab * np.ones((3,)),q_ab * np.ones((3,))))
        self.Qc = np.linalg.cholesky(np.diag(Q0))  # first round cholesky
        self.W = np.eye(3) * (2e-3) ** 2

        # init trajectory
        quat0 = mathutils.Matrix(np.eye(3)).to_quaternion()
        v0 = np.array([0., 0., 0.])
        x0 = np.array([0., 0., 0.])
        u0 = np.zeros((6,))
        self.trajectory = np.concatenate((quat0, v0, x0, u0))

        # init observetime
        self.timestamp = timestamp
        self.obsTime = np.zeros((len(timestamp),))
        self.obsTime[::5] = 1

    def iekfPropagation(self, chi_i, bias_i, P_i, u_i, Q_i, dt_i):
        # IEKF on Lie Groups
        # N_lm = chi_i[:, 5:].shape[1]
        N_P = len(P_i)
        N_Q = len(Q_i)

        # state propagation
        omega_i = u_i[:3]
        acc_i = u_i[3:]
        omega_b = bias_i[:3]
        acc_b = bias_i[3:]

        Rot_i = chi_i[:3, :3] @ self.lg.expSO3((omega_i - omega_b) * dt_i)
        delta_a_i = Rot_i @ (acc_i - acc_b)
        v_i = chi_i[:3, 3] + (delta_a_i + self.g) * dt_i
        x_i = chi_i[:3, 4] + v_i * dt_i

        # covariance propagation
        F_i = np.eye(N_P)
        F_i[:3, 9:12] = - Rot_i * dt_i

        F_i[3:6, :3] = self.lg.hat_operator(self.g) * dt_i
        F_i[3:6, 9:12] = -self.lg.hat_operator(v_i) @ Rot_i * dt_i
        F_i[3:6, 12:15] = -Rot_i * dt_i

        F_i[6:9, :3] = self.lg.hat_operator(self.g) * dt_i * dt_i
        F_i[6:9, 3:6] = np.eye(3) * dt_i
        F_i[6:9, 9:12] = -self.lg.hat_operator(x_i) @ Rot_i * dt_i
        F_i[6:9, 12:15] = -Rot_i * dt_i * dt_i

        # if N_lm > 0:
        #     for i in range(N_lm):
        #         p_i = chi_i[:3, i + 9]
        #         F_i[15 + 3 * i:18 + 3 * i, 9:12] = -self.lg.hat_operator(p_i) @ Rot_i * dt_i

        G_i = np.zeros((N_P, N_Q))
        G_i[:3, :3] = Rot_i
        G_i[:3, 6:9] = Rot_i * dt_i

        G_i[3:6, :3] = self.lg.hat_operator(v_i) * Rot_i
        G_i[3:6, 3:6] = Rot_i
        G_i[3:6, 6:9] = self.lg.hat_operator(v_i) * Rot_i * dt_i * dt_i
        G_i[3:6, 9:12] = Rot_i * dt_i * dt_i

        G_i[6:9, :3] = self.lg.hat_operator(x_i) * Rot_i
        G_i[6:9, 3:6] = Rot_i * dt_i
        G_i[6:9, 6:9] = self.lg.hat_operator(x_i) * Rot_i * dt_i * dt_i * dt_i
        G_i[6:9, 9:12] = Rot_i * dt_i * dt_i * dt_i

        G_i[9:15, 6:12] = np.eye(6)

        P_predict = F_i @ P_i @ F_i.T + G_i @ (Q_i * dt_i) @ G_i.T * dt_i
        chi_predict = self.lg.state2chi(Rot_i, v_i, x_i, None)

        return chi_predict, P_predict

    def rukfObservation(self, chi, n_n):
        # >>> chi, xi, param, n_n
        # Pi = param.Pi
        # chiC = param.chiC
        # RotC = chiC[:3, :3]
        # xC = chiC[:3, 3]
        #
        # yAmers = param.yAmers
        # NbAmers = len(yAmers)
        #
        # chi_j = np.dot(chi, lg.expSE3(xi))
        # Rot = chi_j[:3, :3]
        # x = chi_j[:3, 4]
        # PosAmers = chi_j[:3, 5:]
        # posAmers = PosAmers[:, yAmers]
        #
        # pC = Rot.T @ (posAmers - np.repeat(x, NbAmers, axis=1))
        # z = Pi @ (RotC.T @ pC - np.repeat(xC, NbAmers, axis=1))
        # y = z[:2, :] / z[2, :]
        # y = y + n_n
        y = chi[:3, 4] + n_n

        return y

    def rukfUpdate(self, chi_i, bias_i, S_i, y_i, R_i):
        k = len(y_i)
        l_S = len(S_i)
        l_aug = l_S + k
        l_R = len(R_i)
        # Rc = np.linalg.cholesky(np.kron(np.eye(k), R_i))
        Rc = np.linalg.cholesky(R_i)
        S_aug = block_diag(S_i, Rc)
        bias_update = np.zeros((6,))

        # scaled unsented transform                  <<<<<<<<TODO CHANGE scaled unsented transform
        W0 = 1 - l_aug / 3
        Wj = (1 - W0) / (2 * l_aug)
        gamma = np.sqrt(l_aug / (1 - W0))
        alpha = 1
        beta = 2

        # Compute transformed measurement
        X = gamma * np.hstack((np.zeros((l_aug, 1)), S_aug.T, -S_aug.T))  # sigma-points
        Y = np.zeros((k, 2 * l_aug + 1))
        Y[:k, 0] = self.rukfObservation(chi_i, np.zeros((k,)))

        # Snorm = np.linalg.norm(S_i)

        for j in range(1, 2 * l_aug + 1):
            # xi_j = np.hstack((X[:9, j], X[15:l_S, j]))
            xi_j = X[:9, j]
            v_j = X[l_S:l_aug, j]
            chi_j = self.lg.expSE3(xi_j) @ chi_i
            Y[:k, j] = self.rukfObservation(chi_j, v_j)
            # print(j)

        ybar = W0 * Y[:k, 0] + Wj * np.sum(Y[:k, 1:], axis=1)  # Measurement mean # FIXME become inifity!!!!!!!!
        Y[:k, 0] = np.sqrt(abs(W0 + (1 - alpha ** 2 + beta))) * (Y[:k, 0] - ybar)
        ybarbar = np.repeat(ybar[:, None], 2 * l_aug, axis=1)
        YY = np.sqrt(Wj) * (Y[:k, 1:2 * l_aug + 1] - ybarbar)
        # Y[k:, l_S + 1:l_S + 4] = Rc
        # Y[k:, l_aug + l_S + 1:] = -Rc

        Rs = np.squeeze(np.array(scipy.linalg.qr(YY.T, mode='r')), axis=0)
        Ss = Rs[:k, :k] + R_i
        Sy = cholupdate(Ss, Y[:k, 0], '-')  # Sy'*Sy = Pyy
        Pyy = Sy.T @ Sy
        Pxy = np.zeros((l_S, k))

        for j in range(1, 2 * l_aug + 1):
            padd = Wj * X[:l_S, j][:, None] @ (Y[:, j] - ybar)[None, :]
            Pxy = Pxy + padd

        K = Pxy @ np.linalg.inv(Pyy)  # Gain
        # Knorm = np.linalg.norm(K)
        xibar = K @ (y_i - ybar)

        # bias update
        bias_update[:3] = bias_i[:3] + xibar[9:12]
        bias_update[3:] = bias_i[3:] + xibar[12:15]
        xibar = xibar[:9]

        # Covariance update
        A = K @ Sy.T
        for n in range(k):  # TODO
            S_i = cholupdate(S_i, A[:, n], '-')

        # Update mean state
        chi_next = self.lg.expSE3(xibar) @ chi_i
        JT = self.lg.vec2adjJl(xibar)
        # JTnorm = np.linalg.norm(JT)
        S_i = S_i @ JT

        return chi_next, bias_update, S_i

    def updateTraj(self, traj, chi, u_b):
        Rot, v, x, _p = self.lg.chi2state(chi)
        # x = Rot @ x
        quat = np.array(mathutils.Matrix(Rot).to_quaternion())
        state_rows = np.concatenate((quat, v, x, u_b))
        traj = np.vstack((traj, state_rows))
        return traj

    def run_hybrid_ukf(self, omega, acc, y_mess, test_quat):
        # init all
        t_i = 0
        trajR = self.trajectory
        # Nmax = omega.shape[1]
        # l_mess = y_mess.shape[1] // 10
        S_R = np.copy(self.S0)
        P_R = self.P0
        Qc = np.copy(self.Qc)
        bias_i = np.copy(self.bias0)
        RotR = self.lg.expSO3(self.xi0[:3])
        vR = self.xi0[3:6]
        xR = self.xi0[6:9]
        chiR = self.lg.state2chi(RotR, vR, xR, None)
        errorlist = np.zeros((1, 3))

        for step_i in range(1, self.iter_steps):

            # propagation
            omega_i = omega[:, step_i]
            acc_i = acc[:, step_i]
            dt = self.timestamp[step_i] - self.timestamp[step_i - 1]
            chiR_last = np.copy(chiR)

            u_i = np.hstack((omega_i, acc_i))
            chiR_predict, P_R = self.iekfPropagation(chiR_last, bias_i, P_R, u_i, Qc, dt)
            S_R = np.linalg.cholesky(P_R).T

            # calculate propagation error
            test_rot = mathutils.Quaternion(test_quat[:, step_i])
            test_theta = Rotation.from_matrix(np.array(test_rot.to_matrix()))
            test_theta = test_theta.as_euler('zyx', degrees=True)
            cal_theta = Rotation.from_matrix(chiR_predict[:3, :3])
            cal_theta = cal_theta.as_euler('zyx', degrees=True)
            x_error = np.linalg.norm(y_mess[:, step_i] - chiR_predict[:3, 4])

            if self.ERROR_CHECK:
                temp_error = test_theta - cal_theta # IF ERROR CHECK temp error denotes angle error
            else:
                theta_error = np.linalg.norm(test_theta - cal_theta) # ELSE theta error denotes angle error norm
                temp_error = np.array([step_i, theta_error, x_error]) # ELSE temp error denotes norm error of all

            errorlist = np.vstack((errorlist, temp_error))

            # measurement and update
            if self.obsTime[step_i] == 1:
                chiR, bias_i, S_R = self.rukfUpdate(chiR_predict, bias_i, S_R, y_mess[:, step_i], self.W)
                # RotR, vR, xR, __ = self.lg.chi2state(chiR)
                trajR = self.updateTraj(trajR, chiR, bias_i)
                t_i += 1
            else:
                trajR = self.updateTraj(trajR, chiR_predict, bias_i)
                chiR = chiR_predict

        return trajR, errorlist


if __name__ == '__main__':
    dtt = 0.01
