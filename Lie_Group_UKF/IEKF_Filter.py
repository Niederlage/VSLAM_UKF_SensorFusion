import numpy as np
from scipy.linalg import block_diag
import scipy
from Lie_Group_UKF.LG_Tool import Lie_Group, cholupdate
import mathutils


class IEKF_Filter():

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

    def iekfPropagation(self, chi_i, bias_i, u_i, P_i, Q_i, dt_i):
        # IEKF on Lie Groups
        N_lm = chi_i[:, 5:].shape[1]
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

        if N_lm > 0:
            for i in range(N_lm):
                p_i = chi_i[:3, i + 9]
                F_i[15 + 3 * i:18 + 3 * i, 9:12] = -self.lg.hat_operator(p_i) @ Rot_i * dt_i

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

    def iekfObservation(self, chi, n_n):
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

    def iekfUpdate(self, chi_i, bias_i, P_i, y_i, R_i):
        l_y = len(y_i)
        l_P = len(P_i)
        l_lm = len(chi_i[:, 5:])
        l_R = len(R_i)
        # Rc = np.linalg.cholesky(np.kron(np.eye(k), R_i))

        Rot_i = chi_i[:3, :3]
        x_i = chi_i[:3, 5]
        H_i = np.zeros((l_y, l_P))
        if l_lm > 0:
            lm_i = chi_i[:3, 5:]

        Pnorm = np.linalg.norm(P_i)


        y_predict = np.hstack((y_i, np.array([0,1])))

        S_i = H_i @ P_i @ H_i.T + R_i
        K_i = P_i @ H_i.T @ np.linalg.inv(S_i)
        P_corrected = (np.eye(l_P) - K_i @ H_i) @ P_i

        xibar = K_i * (y_i - y_predict) # innovation
        bias_i[:3]=  bias_i[:3] + xibar[9:12]
        bias_i[3:]=  bias_i[3:] + + xibar[12:15]
        xibar = xibar[:9]

        # Update mean state
        chi_next = self.lg.expSE3(xibar) @ chi_i

        return chi_next, bias_i, P_corrected

    def updateTraj(self, traj, chi, u):
        Rot, v, x, _p = self.lg.chi2state(chi)
        quat = np.array(mathutils.Matrix(Rot).to_quaternion())
        state_rows = np.concatenate((quat, v, x, u))
        traj = np.vstack((traj, state_rows))
        return traj

    def run_iekf(self, omega, acc, y_mess):
        # init all
        t_i = 0
        trajI = self.trajectory
        Nmax = omega.shape[1]
        l_mess = y_mess.shape[1]
        P_I = self.P0
        Qc = self.Qc
        bias_i = self.bias0
        RotI = self.lg.expSO3(self.xi0[:3])
        vI = self.xi0[3:6]
        xI = self.xi0[6:9]
        chiI = self.lg.state2chi(RotI, vI, xI, None)

        for step_i in range(1, l_mess):

            # propagation
            omega_i = omega[:, step_i]
            acc_i = acc[:, step_i]
            dt = self.timestamp[step_i] - self.timestamp[step_i - 1]
            chiI_last = np.copy(chiI)

            # motion dynamic
            dRotI = self.lg.expSO3((omega_i - bias_i[:3]) * dt)
            RotI = RotI @ dRotI
            dvI = (RotI @ (acc_i - bias_i[3:]) + self.g) * dt
            vI = vI + dvI
            xI = xI + vI * dt
            chiI_predict = self.lg.state2chi(RotI, vI, xI, None)

            u_i = np.hstack((omega_i, acc_i))
            chiI_predict, P_I = self.iekfPropagation(chiI_last, bias_i, P_I, u_i, Qc, dt)
            normP_I = np.linalg.norm(P_I)
            # print(t_i)
            # measurement and update
            if self.obsTime[step_i] == 1:
                chiI, bias_i, P_I = self.iekfUpdate(chiI_predict, bias_i, P_I, y_mess[:, t_i], self.W)
                RotI, vI, xI, __ = self.lg.chi2state(chiI)
                trajI = self.updateTraj(trajI, chiI, u_i)
                t_i += 1
            else:
                trajI = self.updateTraj(trajI, chiI_predict, u_i)
                chiI = chiI_predict

            if (t_i > 30):
                break

        return trajI


if __name__ == '__main__':
    dtt = 0.01
