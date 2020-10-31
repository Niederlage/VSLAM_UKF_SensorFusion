import numpy as np
from scipy.linalg import block_diag
import scipy
from Lie_Group_UKF.LG_Tool import Lie_Group, cholupdate
import mathutils
from scipy.spatial.transform import Rotation


class UKF_LG_Right_Filter():

    def __init__(self, xi0, bias0, timestamp):
        self.g = np.array([0, 0, -9.80665])
        self.xi0 = xi0
        self.bias0 = bias0
        self.lg = Lie_Group()

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
            (q_omega * np.ones((3,)), q_a * np.ones((3,)), q_omega * np.ones((3,)), q_omegab * np.ones((3,))))
        self.Qc = np.diag(np.sqrt(Q0))  # first round cholesky
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
        self.obsTime[::10] = 1

    def rukfPropagation(self, chi_post, chi_j, bias_j, S_j, u_j, Q_j, dt_j):
        # Left-UKF on Lie Groups
        l_S = len(S_j)  # state size
        Qc = np.linalg.cholesky(Q_j).T
        S_aug = block_diag(S_j, Qc)
        l_aug = len(S_aug)

        # scaled unsented transform           <<<<<<<<TODO CHANGE scaled unsented transform
        W0 = 1 - l_aug / 3
        Wj = (1 - W0) / (2 * l_aug)
        gamma = np.sqrt(l_aug / (1 - W0))
        v_0 = chi_j[:3, 3]
        x_0 = chi_j[:3, 4]
        R_0 = chi_j[:3, :3]

        # Prediction
        ichi = self.lg.invSE3(chi_post)
        u_j -= bias_j  # unbiased inputs
        X = gamma * np.hstack((np.zeros((l_aug, 1)), S_aug, -S_aug))  # sigma-points
        d_bias = X[l_S + 6:l_S + 12, :] * dt_j
        X[9:15, :] = X[9:15, :] + d_bias  # add bias noise

        for j in range(1, 2 * l_aug + 1):
            # xi_j = np.hstack((X[:9, j], X[15:l_S, j]))  # do not take bias

            # select sample incremental state
            xi_j = X[:9, j]  # do not take bias
            w_j = X[l_S:l_aug, j]
            omega_bj = X[9:12, j]
            a_bj = X[12:15, j]

            # nonlinear- Lie Group propagation
            chi_j = self.lg.expSE3(xi_j) @ chi_j
            Rot_j = R_0 @ self.lg.expSO3((u_j[:3] - omega_bj + w_j[:3]) * dt_j)
            theta_j = Rotation.from_matrix(Rot_j)
            theta_ = theta_j.as_euler('zyx', degrees=True)
            da_imu = u_j[3:] - a_bj + w_j[3:6]
            da_j = Rot_j @ da_imu
            dv_inertial = (da_j + self.g) * dt_j
            v_j = v_0 + dv_inertial
            x_j = x_0 + v_j * dt_j
            chi = self.lg.state2chi(Rot_j, v_j, x_j, chi_j[:3, 5:])
            Xi_j = chi @ ichi  # can be more time efficient
            logXi_j = self.lg.logSE3(Xi_j)
            X[:9, j] = logXi_j[:9]  # propagated sigma points
            # X[16 - 1:l_S, j - 1] = logXi_j[16 - 1:l_S, j - 1]

        X = np.sqrt(Wj) * X
        Rs = np.squeeze(scipy.linalg.qr(X[:l_S, 1:2 * l_aug + 1].T, mode='r'))
        S_propagate = Rs[:l_S, :l_S]
        normS_propagate = np.linalg.norm(S_propagate)
        return S_propagate

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

        Snorm = np.linalg.norm(S_i)

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
        Ss = Rs[:k, :k]
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
        bias_i[:3] = bias_i[:3] + xibar[9:12]
        bias_i[3:] = bias_i[3:] + xibar[12:15]
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

        return chi_next, bias_i, S_i

    def updateTraj(self, traj, chi, u):
        Rot, v, x, _p = self.lg.chi2state(chi)
        # x = Rot @ x
        quat = np.array(mathutils.Matrix(Rot).to_quaternion())
        state_rows = np.concatenate((quat, v, x, u))
        traj = np.vstack((traj, state_rows))
        return traj

    def run_ukf(self, omega, acc, y_mess, test_quat):
        # init all
        t_i = 0
        trajR = self.trajectory
        Nmax = omega.shape[1]
        l_mess = y_mess.shape[1] // 10
        S_R = np.copy(self.S0)
        Qc = np.copy(self.Qc)
        bias_i = np.copy(self.bias0)
        RotR = self.lg.expSO3(self.xi0[:3])
        vR = self.xi0[3:6]
        xR = self.xi0[6:9]
        chiR = self.lg.state2chi(RotR, vR, xR, None)
        errorlist = np.zeros((1, 3))

        for step_i in range(1, 3000):

            # propagation
            omega_i = omega[:, step_i]
            acc_i = acc[:, step_i]
            dt = self.timestamp[step_i] - self.timestamp[step_i - 1]
            chiR_last = np.copy(chiR)

            # motion dynamic
            dRotR = self.lg.expSO3((omega_i - bias_i[:3]) * dt)
            RotR = RotR @ dRotR
            dvR = (RotR @ (acc_i - bias_i[3:]) + self.g) * dt
            vR = vR + dvR
            xR = xR + vR * dt
            chiR_predict = self.lg.state2chi(RotR, vR, xR, None)

            # print(xR)

            u_i = np.hstack((omega_i, acc_i))
            S_R = self.rukfPropagation(chiR_last, chiR_predict, bias_i, S_R, u_i, Qc, dt)

            # calculate propagation error
            test_rot = mathutils.Quaternion(test_quat[:, step_i])
            test_rot = test_rot.to_matrix()
            test_theta = Rotation.from_matrix(np.array(test_rot))
            test_theta = test_theta.as_euler('zyx', degrees=True)
            cal_theta = Rotation.from_matrix(chiR_predict[:3, :3])
            cal_theta = cal_theta.as_euler('zyx', degrees=True)

            x_error = np.linalg.norm(y_mess[:, step_i] - chiR_predict[:3, 4])
            theta_error = np.linalg.norm(test_theta - cal_theta)
            temp_error = np.array([step_i, theta_error, x_error])
            errorlist = np.vstack((errorlist, temp_error))

            # print(t_i)
            # measurement and update
            if self.obsTime[step_i] == 1:
                chiR, bias_i, S_R = self.rukfUpdate(chiR_predict, bias_i, S_R, y_mess[:, step_i], self.W)
                RotR, vR, xR, __ = self.lg.chi2state(chiR)
                trajR = self.updateTraj(trajR, chiR, u_i)
                t_i += 1
            else:
                trajR = self.updateTraj(trajR, chiR_predict, u_i)
                chiR = chiR_predict

            # if (t_i % 20 == 0):
            #     print(y_mess[:, step_i] - chiR_predict[:3, 4])
            #     print(t_i)
            # if theta_error > 10:
            #     print("theta error exceed!")
                # print("theta error:",test_theta)
                # print(y_mess[:, step_i] - chiR_predict[:3, 4])
                # print(t_i)

        return trajR, errorlist


if __name__ == '__main__':
    dtt = 0.01
