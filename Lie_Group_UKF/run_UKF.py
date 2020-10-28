# @INPROCEEDINGS{2018_Brossard_Invariant,
#   author = {Martin Brossard, Silvère Bonnabel, and Axel Barrau},
#   booktitle={2018 21st International Conference on Information Fusion (FUSION)},
#   title={Invariant Kalman Filtering for Visual Inertial SLAM},
#   year={2018},
#   pages={2021-2028},
#   doi={10.23919/ICIF.2018.8455807},
#   month={July},}

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Lie_Group_UKF.UKF_Filter_Right import UKF_LG_Right_Filter
from Lie_Group_UKF.LG_Tool import Lie_Group
import mathutils


def plot_pose(pose_w, pose_real, display3D=False):
    fig = plt.figure()

    if display3D:
        ax = fig.gca(projection='3d')
        ax.plot(pose_real[:, 0], pose_real[:, 1], pose_real[:, 2], "-", label='traj real')
        ax.plot(pose_w[:, 0], pose_w[:, 1], pose_w[:, 2], "-", label='traj UKF right')
        ax.scatter(pose_w[0, 0], pose_w[0, 1], pose_w[0, 2], color="red")
        ax.scatter(pose_w[100, 0], pose_w[100, 1], pose_w[10, 2], color="green")
        # ax.scatter(pose_w[-1, 0], pose_w[-1, 1], pose_w[-1, 2], color="purple")

        ax.set_zlabel('Z')
    else:
        ax = fig.gca()
        ax.plot(pose_w[:, 0], pose_w[:, 1], "o-", label='pose points')

        ax.plot(pose_w[0, 0], pose_w[0, 1], "X", color="red")
        ax.plot(pose_w[10, 0], pose_w[10, 1], "X", color="green")
        ax.plot(pose_w[-1, 0], pose_w[-1, 1], "X", color="purple")

    ax.legend()  # 画一条空间曲线
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_xlim(-5, 12)
    # ax.set_ylim(-6, 12)
    # ax.set_zlim(-6, 6)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    lg = Lie_Group()

    # load test data
    # trajR = loadmat("data/trajR.mat")['trajR']
    # rot_trajR = trajR[0][0][0]
    # v_trajR = trajR[0][0][4]
    # x_trajR = trajR[0][0][5]
    # omega_b_trajR = trajR[0][0][6]
    # a_b_trajR = trajR[0][0][7]

    trajReal = loadmat("data/trajReal2244.mat")['trajReal']
    a_b_trajReal = loadmat("data/trajReal_a_b.mat")['trajReal'][0][0][0]
    omega_b_trajReal = loadmat("data/trajReal_omega_b.mat")['trajReal'][0][0][0]
    quat_trajReal = loadmat("data/trajReal_quat.mat")['trajReal'][0][0][0]
    x_trajReal = trajReal[0][0][3]
    v_trajReal = trajReal[0][0][4]

    # correcting starting IMU time
    # tmin = 1081-1
    omega_input = loadmat("data/omega.mat")["omega"]
    acc_input = loadmat("data/acc.mat")["acc"]
    y_measure = x_trajReal[:, ::10]
    errorR = loadmat("data/errorR.mat")['errorR']
    errorR_errorR = errorR[0][0][0]
    errorX_errorR = errorR[0][0][1]

    tIMU = np.array(loadmat('data/tIMU.mat')['tIMU']).flatten() * 1e-9
    quat0 = np.array([0.132045000000000, 0.803580000000000, -0.189041000000000, 0.548715000000000])
    rot_from_quat = mathutils.Quaternion(quat0).to_matrix()
    # ref_Rot0 = np.array([[0.326351607162659, -0.448728672426492, 0.831947839136990],
    #                      [-0.158908774789464, -0.893655380082092, -0.419676140547974],
    #                      [0.931795379789609, 0.00475817114965219, -0.362952793087544]])
    # ref_theta0 = lg.logSO3(ref_Rot0) # [ 2.3321971  -0.54864588  1.59251286]
    # theta_trajReal = lg.logSO3(np.array(rot_trajReal)) # [ 1.59244725 -0.5486233   2.33210082]
    theta_trajReal = lg.logSO3(np.array(rot_from_quat))  # [ 2.3321971  -0.54864588  1.59251286]

    xi0 = np.hstack((theta_trajReal, v_trajReal[:, 0], x_trajReal[:, 0]))
    omega_b0 = np.array([-0.00252022798492071, 0.0213195611407080, 0.0777716961173257])
    acc_b0 = np.array([-0.0120978472707349, 0.105159897404514, 0.0904481027396282])
    bias0 = np.hstack((omega_b0, acc_b0))

    # run UKF filter
    ukf_right = UKF_LG_Right_Filter(xi0, bias0, tIMU)
    test_traj = ukf_right.run_ukf(omega_input, acc_input, y_measure)

    # plot_pose(x_trajR.T, x_trajReal.T, display3D=True)
print(1)
