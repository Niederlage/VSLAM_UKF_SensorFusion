import numpy as np
import matplotlib.pyplot as plt
from mathutils import Quaternion
from blender_to_traj import Blender_to_Traj
from msg_to_traj import Msg_to_Traj


def get_corresponding_frame(b_list, m_list, skip_fr=3):
    framelist = np.array(skip_fr * m_list[:, 1], dtype=int)
    return b_list[framelist, 1:3]


def cal_scalar_error(br_list, mr_list):
    skalar = br_list / mr_list
    skalar = np.delete(skalar, 0, 0)
    return skalar, np.average(skalar)


def plot_baseline(b_line, m_line):
    fig = plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    # plot ground truth baseline info
    ax = fig.add_subplot(221)
    ax.plot(b_line, color='tab:red', label='ground truth')
    # ax.set_xlabel('frame', fontsize=16)
    ax.set_ylabel('baseline/m', fontsize=16)
    ax.set_title('Ground Truth', fontsize=16)
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', rotation=0)
    ax.grid(alpha=.4)
    ax.grid(True)

    # # # Plot Line2 (Right Y Axis)
    ax = fig.add_subplot(222)
    counts, bins = np.histogram(b_line)
    # edges = np.delete(bins, 0)
    # res = edges.dot(counts) / len(b_line)
    b_sum = 0
    for i, _ in enumerate(b_line):
        b_sum += b_line[i] ** 2
    b_RMSE = np.sqrt(b_sum / len(b_line))

    ax.hist(bins[:-1], bins, weights=counts, facecolor='tab:red', alpha=0.75)
    b_str = 'Groud Truth RPE:={:.3f}m'.format(b_RMSE)
    plt.text(0.42, 400, b_str, fontsize=16)
    # ax.grid(alpha=.4)
    # ax.grid(True)

    # # # ax (right Y axis)
    ax.set_ylabel("frequency", fontsize=16)
    ax.tick_params(axis='y')  # , labelcolor='tab:blue'
    # ax.set_xlabel('baseline/m', fontsize=16)
    # plt.ylim(0, 1)
    ax.set_title("Distribution of Baseline Length", fontsize=16)

    # plot OpenVSLAM baseline info
    ax = fig.add_subplot(223)
    ax.plot(m_line, color='tab:blue', label='OpenVSLAM')
    ax.set_xlabel('frame', fontsize=16)
    ax.set_ylabel('baseline/m', fontsize=16)
    ax.set_title('OpenVSLAM', fontsize=16)
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', rotation=0)
    ax.grid(alpha=.4)
    ax.grid(True)

    # # # Plot Line2 (Right Y Axis)
    ax = fig.add_subplot(224)
    counts, bins = np.histogram(m_line)
    # edges = np.delete(bins, 0)
    # res = edges.dot(counts) / len(m_line)
    m_sum = 0
    for i, _ in enumerate(m_line):
        m_sum += m_line[i] ** 2
    m_RMSE = np.sqrt(m_sum / len(m_line))
    ax.hist(bins[:-1], bins, weights=counts, facecolor='tab:blue', alpha=0.75)
    m_str = 'OpenVSLAM RPE :={:.3f}m'.format(m_RMSE)
    plt.text(0.8, 35, m_str, fontsize=16)
    # ax.grid(alpha=.4)
    # ax.grid(True)

    # # # ax (right Y axis)
    ax.set_ylabel("frequency", fontsize=16)
    ax.tick_params(axis='y')
    ax.set_xlabel('baseline/m', fontsize=16)
    # plt.ylim(0, 1)
    # ax.set_title("Distribution of Baseline", fontsize=22)

    # plt.tight_layout()
    # plt.show()


def plot_traj(pose_b, pose_m):
    fig = plt.figure()

    ax = fig.gca()
    # draw groud truth trajectory
    ax.plot(pose_b[:, 0], pose_b[:, 1], "-", color='tab:red', label='groud truth trajectory')
    ax.plot(pose_b[0, 0], pose_b[0, 1], color="red")
    # ax.scatter(pose_b[10, 0], pose_b[10, 1], color="green")
    ax.scatter(pose_b[-1, 0], pose_b[-1, 1], color="purple")

    # draw OpenVSLAM trajectory
    ax.plot(pose_m[:, 0], pose_m[:, 1], "-", color='tab:blue', label='OpenVSLAM trajectory')
    ax.plot(pose_m[0, 0], pose_m[0, 1], color="red")
    # ax.scatter(pose_m[10, 0], pose_m[10, 1], color="green")
    ax.scatter(pose_m[-1, 0], pose_m[-1, 1], color="purple")

    ax.legend()  # 画一条空间曲线
    ax.set_xlabel('X/m', fontsize=16)
    ax.set_ylabel('Y/m', fontsize=16)

    # plt.show()


def plot_all(traj_b, traj_m, baseline_b, baseline_m, scalar_list, ave):
    fig = plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    ###################### plot ground truth baseline info #####################################
    ax = fig.add_subplot(231)
    ax.plot(baseline_b, color='tab:red', label='ground truth')
    ax.set_ylabel('baseline/m', fontsize=12)
    ax.set_title('Ground Truth', fontsize=16)
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', rotation=0)
    ax.grid(alpha=.4)
    ax.grid(True)

    ax.set_ylim(0, 8)  # skip3 16
    b_sum = 0
    for i, _ in enumerate(baseline_b):
        b_sum += baseline_b[i] ** 2
    b_RMSE = np.sqrt(b_sum / len(baseline_b))
    b_str = 'Groud Truth baseline:{:.3f}m'.format(b_RMSE)
    plt.text(0, 7, b_str, fontsize=14) # skip3 14

    ##################### plot groud truth distribution ########################################
    ax = fig.add_subplot(232)
    counts, bins = np.histogram(baseline_b, bins=30)
    ax.hist(bins[:-1], bins, weights=counts, facecolor='tab:red', alpha=0.75)
    ax.set_ylabel("frequency", fontsize=12)
    ax.tick_params(axis='y')  # , labelcolor='tab:blue'
    ax.set_title("Distribution of Baseline Length", fontsize=16)

    ##################### draw groud truth trajectory ########################################
    ax = fig.add_subplot(233)
    ax.plot(traj_b[:, 0], traj_b[:, 1], "-", color='tab:red', label='groud truth trajectory')
    ax.plot(traj_b[0, 0], traj_b[0, 1], color="red")
    # ax.scatter(traj_b[10, 0], traj_b[10, 1], color="green")
    ax.scatter(traj_b[-1, 0], traj_b[-1, 1], color="purple")

    # draw OpenVSLAM trajectory
    ax.plot(traj_m[:, 0], traj_m[:, 1], "-", color='tab:blue', label='OpenVSLAM trajectory')
    ax.plot(traj_m[0, 0], traj_m[0, 1], color="red")
    ax.scatter(traj_m[-1, 0], traj_m[-1, 1], color="purple")
    ax.set_title('trajectory compare',fontsize=16)
    ax.legend(fontsize=14)  # 画一条空间曲线
    ax.set_xlabel('X/m', fontsize=10)
    ax.set_ylabel('Y/m', fontsize=10)

    ##################### plot OpenVSLAM baseline info ########################################
    ax = fig.add_subplot(234)
    ax.plot(baseline_m, color='tab:blue', label='OpenVSLAM')
    ax.set_xlabel('frame', fontsize=12)
    ax.set_ylabel('baseline/m', fontsize=12)
    ax.set_title('OpenVSLAM', fontsize=16)
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', rotation=0)
    ax.grid(alpha=.4)
    ax.grid(True)
    m_sum = 0
    for i, _ in enumerate(baseline_m):
        m_sum += baseline_m[i] ** 2
    m_RMSE = np.sqrt(m_sum / len(baseline_m))
    m_str = 'OpenVSLAM baseline:{:.3f}m'.format(m_RMSE)
    plt.text(0, 0.75, m_str, fontsize=14)  #skip3 1.75
    ax.set_ylim(0, 0.9)  # skip3 1.95
    ##################### draw groud truth trajectory ########################################
    ax = fig.add_subplot(235)
    counts, bins = np.histogram(baseline_m, bins=30)
    ax.hist(bins[:-1], bins, weights=counts, facecolor='tab:blue', alpha=0.75)
    ax.set_ylabel("frequency", fontsize=12)
    ax.tick_params(axis='y')
    ax.set_xlabel('baseline/m', fontsize=12)

    ############################### draw scalar error ########################################
    ax = fig.add_subplot(236)
    line = np.ones((len(scalar_list),1)) * ave
    ax.plot(scalar_list, 'g+')
    ax.plot(line, '_')
    ax.set_title('scalar compare', fontsize=16)
    ax.set_xlabel("keyframe", fontsize=12)
    s_str = 'scalar error mean:{:.3f}'.format(ave)
    plt.text(0, 9.1, s_str, fontsize=14) # skip3 9
    ax.set_ylim(8.4, 9.3)  # skip3 0,10


def main():
    # msgpath = 'test_simu_equirectangular_resident.msg'
    # msgpath = 'resident_EQT_0p5.msg'
    msgpath = 'resident_EQT_0p5_skip0_kp2000_noloop.msg'
    # msgpath = 'uni_EQT_1200_1p05.msg'
    blenderpath = 'Camera_Resident_ideal_blender.csv'
    # blenderpath = 'Camera_Uni_ideal_blender.csv'
    msgfolderpath = 'saved_data/from_msg_data/'
    blenderfolderpath = 'saved_data/from_blender_csv/'

    msgtraj = Msg_to_Traj(msgpath)
    blendertraj = Blender_to_Traj(blenderfolderpath + blenderpath)

    # loading msg data
    if LOAD_MSGDATA:
        load_data = np.load(msgfolderpath + 'saved_resident_EQT_0p5_skip0_kp2000_noloop.npz')
        print('file loaded successfully...')
        kf_pose_cw = load_data['keyframe_pose_cw']

    else:
        landmarks, keyframe_scale, kf_pose_cw, keyframe_undists = msgtraj.msg_unpack_to_array()

    m_pose = kf_pose_cw[kf_pose_cw[:, 0].argsort()]  # 按第'1'列排序

    m_traj = msgtraj.get_trajectory(m_pose[:, 2:])
    m_baseline = msgtraj.cal_baseline(m_traj)

    # adjust map orientation
    m_traj = m_traj[:, ::2]  # (82, 2)
    m_traj[:, 1] = -m_traj[:, 1]
    m_traj = blendertraj.rot_traj(m_traj, np.pi * 2 / 4)  # resident pi * 2/4 #
    # loading blender data
    b_pose = blendertraj.csv_to_pose()
    b_traj = b_pose[:, 1:3]
    b_traj = blendertraj.rot_traj(b_traj, -np.pi * 0 / 4)  # resident pi * 0/4 #
    #b_baseline = blendertraj.cal_baseline(b_pose[:, 1:4])
    start_bias = b_traj[0, :]

    # calculate scalar koefficient
    b_traj_cp = get_corresponding_frame(b_pose, m_pose, skip_fr=1)
    b_traj_cp = blendertraj.rot_traj(b_traj_cp - start_bias, np.pi * 0 / 4)   # resident pi * 0/4 #
    br_baseline = blendertraj.cal_baseline(b_traj_cp)
    scalar, ave_val = cal_scalar_error(br_baseline, m_baseline)

    if PLOT_TRAJ:
        plot_traj(b_traj_cp, m_traj)

    if PLOT_BASELINE:
        plot_baseline(br_baseline, m_baseline)
        # plot_baseline(b_baseline, m_baseline)
    if PLOT_ALL:
        plot_all(b_traj_cp, m_traj, br_baseline, m_baseline, scalar, ave_val)
    plt.show()


if __name__ == '__main__':
    LOAD_MSGDATA = True
    PLOT_TRAJ = True
    PLOT_BASELINE = False
    PLOT_ALL = False
    main()
