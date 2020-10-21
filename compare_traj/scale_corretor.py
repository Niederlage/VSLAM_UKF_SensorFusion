import numpy as np
import matplotlib.pyplot as plt
from mathutils import Quaternion
from compare_traj.blender_to_traj import Blender_to_Traj
from compare_traj.msg_to_traj import Msg_to_Traj


def cal_scalar_error(br_list, mr_list):
    skalar = br_list / mr_list
    skalar = np.delete(skalar, 0, 0)
    return skalar, np.average(skalar)


def scale_corrector(pose_vslam, pose_gps):
    window = 10
    steps = len(pose_gps)
    i = 0
    pose_corrected = np.copy(pose_vslam)
    opt_scale = 1
    while ((window + i) < steps):
        x_mat = pose_vslam[i:window + i]
        y_mat = pose_gps[i:window + i]

        sigmax2 = get_sigma(x_mat)
        sigmay2 = get_sigma(y_mat)
        sigmaxy = np.sqrt(sigmax2 * sigmay2)

        sxx = sigmay2 * np.sum(x_mat ** 2)
        syy = sigmax2 * np.sum(y_mat ** 2)
        sxy = sigmaxy * np.trace(x_mat.T @ y_mat)

        subsxy = sxx - syy
        opt_scale = subsxy + np.sign(sxy) * np.sqrt(subsxy ** 2 + 4 * sxy ** 2)
        opt_scale = opt_scale * 0.5 * np.sqrt(sigmax2 / sigmay2) / sxy
        scale_norm = opt_scale ** 2 * sigmay2 + sigmax2
        pose_corrected[i] = (opt_scale * sigmay2 * pose_vslam[i] + sigmax2 * pose_gps[i]) / scale_norm
        # pose_corrected[i] *=opt_scale
        i += 1

    # pose_corrected[-window:] = opt_scale * pose_vslam[-window:]

    return pose_corrected


def get_sigma(sample):
    l_sample = len(sample)
    sigma_mu2 = 0
    for i in range(1, l_sample - 1):
        sigma_mu2 += np.sum((sample[i - 1] - 2 * sample[i] + sample[i + 1]) ** 2)

    return sigma_mu2 / ((l_sample - 3) * 3)


def get_corresponding_frame(b_list, m_list, skip_fr=3):
    framelist = np.array(skip_fr * m_list[:, 1], dtype=int)
    return b_list[framelist, 1:3]


def cal_Traj_RMSE(traj_sample, traj_ref):
    traj_diff = traj_ref - traj_sample
    normlist = np.linalg.norm(traj_diff, axis=1)
    return np.sqrt(np.sum(normlist ** 2) / len(traj_diff)), normlist


def plot_traj(traj_b, traj_m, traj_scaled):
    fig = plt.figure()
    ###################### plot all traj #####################################
    ax = fig.add_subplot(111)
    ax.plot(traj_b[:, 0], traj_b[:, 1], "-", color='g', label='reference trajectory')
    ax.plot(traj_b[0, 0], traj_b[0, 1], "v", color="g")
    # ax.scatter(traj_b[10, 0], traj_b[10, 1], color="green")
    ax.plot(traj_b[-1, 0], traj_b[-1, 1], "^", color="g")

    # ax.plot(traj_noise[:, 0], traj_noise[:, 1], ".", color='tab:red', label='groud truth trajectory')
    # ax.plot(traj_noise[0, 0], traj_noise[0, 1], "v", color="tab:red")
    # # ax.scatter(traj_b[10, 0], traj_b[10, 1], color="green")
    # ax.plot(traj_noise[-1, 0], traj_noise[-1, 1], "^", color="tab:red")

    # draw OpenVSLAM trajectory
    ax.plot(traj_m[:, 0], traj_m[:, 1], "-", color='tab:blue', label='OpenVSLAM trajectory')
    ax.plot(traj_m[0, 0], traj_m[0, 1], "v", color="tab:blue")
    # ax.scatter(traj_m[10, 0], traj_m[10, 1], color="green")
    ax.plot(traj_m[-1, 0], traj_m[-1, 1], "^", color="tab:blue")

    # draw scaled trajectory
    ax.plot(traj_scaled[:, 0], traj_scaled[:, 1], "-", color='#663399', label='scaled trajectory')
    ax.plot(traj_scaled[0, 0], traj_scaled[0, 1], "v", color="#663399")
    # ax.scatter(traj_b[10, 0], traj_b[10, 1], color="green")
    ax.plot(traj_scaled[-1, 0], traj_scaled[-1, 1], "^", color="#663399")

    ax.legend()  # 画一条空间曲线
    ax.set_xlabel('X/m', fontsize=16)
    ax.set_ylabel('Y/m', fontsize=16)
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', rotation=0, labelsize=12)
    # ax.set_ylim(0,6)
    ax.grid(alpha=.4)
    ax.grid(True)


def plot_RMSE(traj_b, traj_noise, traj_m, traj_scaled):
    RMSE_before, norm_before = cal_Traj_RMSE(traj_m, traj_b)
    RMSE_after, norm_after = cal_Traj_RMSE(traj_scaled, traj_b)
    RMSE_ref, norm_ref = cal_Traj_RMSE(traj_noise, traj_b)
    fig = plt.figure()

    # ###################### plot all traj #####################################
    ax = fig.add_subplot(111)
    ax.plot(traj_b[:, 0], traj_b[:, 1], "-", color='g', label='reference trajectory')
    ax.plot(traj_b[0, 0], traj_b[0, 1],"v", color="g")
    # ax.scatter(traj_b[10, 0], traj_b[10, 1], color="green")
    ax.plot(traj_b[-1, 0], traj_b[-1, 1],"^", color="g")

    ax.plot(traj_noise[:, 0], traj_noise[:, 1], ".", color='tab:red', label='groud truth trajectory')
    ax.plot(traj_noise[0, 0], traj_noise[0, 1], "v",color="tab:red")
    # ax.scatter(traj_b[10, 0], traj_b[10, 1], color="green")
    ax.plot(traj_noise[-1, 0], traj_noise[-1, 1],"^", color="tab:red")

    # draw OpenVSLAM trajectory
    ax.plot(traj_m[:, 0], traj_m[:, 1], "-", color='tab:blue', label='OpenVSLAM trajectory')
    ax.plot(traj_m[0, 0], traj_m[0, 1], "v", color="tab:blue")
    # ax.scatter(traj_m[10, 0], traj_m[10, 1], color="green")
    ax.plot(traj_m[-1, 0], traj_m[-1, 1], "^", color="tab:blue")

    # draw scaled trajectory
    ax.plot(traj_scaled[:, 0], traj_scaled[:, 1], "-", color='#663399', label='scaled trajectory')
    ax.plot(traj_scaled[0, 0], traj_scaled[0, 1], "v", color="#663399")
    # ax.scatter(traj_b[10, 0], traj_b[10, 1], color="green")
    ax.plot(traj_scaled[-1, 0], traj_scaled[-1, 1], "^", color="#663399")

    ax.legend()  # 画一条空间曲线
    ax.set_xlabel('X/m', fontsize=16)
    ax.set_ylabel('Y/m', fontsize=16)
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', rotation=0, labelsize=12)
    # ax.set_ylim(0,6)
    ax.grid(alpha=.4)
    ax.grid(True)

    # # # ###################### plot RMSE #####################################
    # ax = fig.add_subplot(111)
    # ax.plot(norm_ref, color='g', label='ground truth-ref error pro frame')
    # ax.plot(norm_before, color='tab:blue', label='OpenVSLAM-ref error pro frame')
    # ax.plot(norm_after, color='#663399', label='corrected OpenVSLAM-ref error pro frame')
    #
    # # bn_str = 'ground truth-ref RMSE :{:.3f}m'.format(RMSE_ref)
    # # plt.text(50, 5.6, bn_str, fontsize=14, color='g')  # skip3 14
    # # bm_str = 'OpenVSLAM-ref RMSE :{:.3f}m'.format(RMSE_before)
    # # plt.text(50, 5.1, bm_str, fontsize=14, color='tab:blue')  # skip3 14
    # # bc_str = 'corrected OpenVSLAM RMSE :{:.3f}m'.format(RMSE_after)
    # # plt.text(50, 4.6, bc_str, fontsize=14, color="#663399")  # skip3 14
    #
    # ax.set_xlabel('keyframe', fontsize=12)
    # ax.set_ylabel('RMSE/m', fontsize=12)
    # ax.set_title('RMSE compare', fontsize=16)
    # ax.tick_params(axis='x', rotation=0, labelsize=12)
    # ax.tick_params(axis='y', rotation=0, labelsize=12)
    # # ax.set_ylim(0,6)
    # ax.grid(alpha=.4)
    # ax.grid(True)
    plt.show()

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

    # ax.set_ylim(0, 8)  # skip3 16
    b_sum = 0
    for i, _ in enumerate(baseline_b):
        b_sum += baseline_b[i] ** 2
    b_RMSE = np.sqrt(b_sum / len(baseline_b))
    b_str = 'Groud Truth baseline:{:.3f}m'.format(b_RMSE)
    plt.text(0, 7, b_str, fontsize=14)  # skip3 14

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
    ax.plot(traj_b[0, 0], traj_b[0, 1], "v", color="tab:red")
    # ax.scatter(traj_b[10, 0], traj_b[10, 1], color="green")
    ax.plot(traj_b[-1, 0], traj_b[-1, 1], "^", color="tab:red")

    # draw OpenVSLAM trajectory
    ax.plot(traj_m[:, 0], traj_m[:, 1], "-", color='tab:blue', label='corrected OpenVSLAM trajectory')
    ax.plot(traj_m[0, 0], traj_m[0, 1], "v", color="tab:blue")
    # ax.scatter(traj_m[10, 0], traj_m[10, 1], color="green")
    ax.plot(traj_m[-1, 0], traj_m[-1, 1], "^", color="tab:blue")

    ax.set_title('trajectory compare', fontsize=16)
    ax.legend(fontsize=14)
    ax.set_xlabel('X/m', fontsize=10)
    ax.set_ylabel('Y/m', fontsize=10)
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', rotation=0, labelsize=12)
    # ax.set_ylim(0,6)
    ax.grid(alpha=.4)
    ax.grid(True)

    ##################### plot corrected baseline info ########################################
    ax = fig.add_subplot(234)
    ax.plot(baseline_m, color='tab:blue', label='corrected OpenVSLAM')
    ax.set_xlabel('frame', fontsize=12)
    ax.set_ylabel('baseline/m', fontsize=12)
    ax.set_title('corrected OpenVSLAM', fontsize=16)
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', rotation=0)
    ax.grid(alpha=.4)
    ax.grid(True)
    m_sum = 0
    for i, _ in enumerate(baseline_m):
        m_sum += baseline_m[i] ** 2
    m_RMSE = np.sqrt(m_sum / len(baseline_m))
    m_str = 'corrected OpenVSLAM baseline:{:.3f}m'.format(m_RMSE)
    plt.text(0, 0.75, m_str, fontsize=14)  # skip3 1.75
    # ax.set_ylim(0, 7)  # skip3 1.95

    ##################### draw corrected distribution ########################################
    ax = fig.add_subplot(235)
    counts, bins = np.histogram(baseline_m, bins=80)
    ax.hist(bins[:-1], bins, weights=counts, facecolor='tab:blue', alpha=0.75)
    ax.set_ylabel("frequency", fontsize=12)
    ax.tick_params(axis='y')
    ax.set_xlabel('baseline/m', fontsize=12)
    ax.set_xlim(0, 7)
    ############################### draw scalar error ########################################
    ax = fig.add_subplot(236)
    line = np.ones((len(scalar_list), 1)) * ave / ave
    ax.plot(scalar_list, 'g+')
    ax.plot(line, '_')
    ax.set_title('scalar compare', fontsize=16)
    ax.set_xlabel("keyframe", fontsize=12)
    s_str = 'scalar error mean:{:.3f}'.format(ave)
    plt.text(0, 1.5, s_str, fontsize=14)  # skip3 9
    ax.set_ylim(0, 3)  # skip3 0,10


def main():
    if USE_RESIDENT:
        msgpath = 'resident_EQT_0p5_skip0_kp2000_noloop.msg'
        blenderpath = "Camera_Resident_noisy_blender.csv"
        # blenderpath = 'Camera_Resident_ideal_blender.csv'
    if USE_UNI:
        blenderpath = 'Camera_Uni_ideal_blender.csv'
        msgpath = 'uni_EQT_2400_skip0_kp2000_noloop_s1p2.msg'

    msgfolderpath = 'saved_data/from_msg_data/'
    blenderfolderpath = 'saved_data/from_blender_csv/'

    msgtraj = Msg_to_Traj(msgpath)
    blendertraj = Blender_to_Traj(blenderfolderpath + blenderpath)

    # loading msg data
    if LOAD_MSGDATA:
        if USE_RESIDENT:
            load_data = np.load(msgfolderpath + 'saved_resident_EQT_0p5_skip0_kp2000_noloop.npz')
        elif USE_UNI:
            # load_data = np.load(msgfolderpath + 'saved_uni_EQT_1200_1p05_skip0_kp4000_loop_s1p9.npz')
            load_data = np.load(msgfolderpath + 'saved_uni_EQT_2400_skip0_kp2000_noloop_s1p2.npz')
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
    if USE_RESIDENT:
        m_traj = blendertraj.rot_traj(m_traj, np.pi * 3.9 / 8)  # resident pi * 2/4 #

    if USE_UNI:
        m_traj = blendertraj.rot_traj(m_traj, np.pi * 0 / 4)  # resident pi * 2/4 #
    # loading blender data
    b_pose = blendertraj.csv_to_pose()
    if "noisy" in blenderpath:
        b_pose[:, [2, 3]] = b_pose[:, [3, 2]]

    b_traj = b_pose[:, 1:3]
    b_baseline = blendertraj.cal_baseline(b_pose[:, 1:4])
    start_bias = b_traj[0, :]

    # calculate scalar koefficient
    if USE_RESIDENT:
        b_traj_cp = get_corresponding_frame(b_pose, m_pose, skip_fr=1) - start_bias  # resident

    elif USE_UNI:
        b_traj_cp = get_corresponding_frame(b_pose, m_pose, skip_fr=1) - start_bias  # uni
        b_traj_cp = blendertraj.rot_traj(b_traj_cp, -np.pi * 5 / 4)  # uni -pi * 5/4

    # add noise
    if ADD_NOISE:
        b_traj_cp_noise = b_traj_cp + 1.5 * np.random.randn(b_traj_cp.shape[0], b_traj_cp.shape[1])  # randn
        # b_traj_cp[:, 0] += 1.8 * np.random.gamma(2, 1., b_traj_cp.shape[0])  # gamma
        # b_traj_cp[:, 1] += 0.3 * np.random.gamma(1, 1.9, b_traj_cp.shape[0])  # gamma

    br_baseline = blendertraj.cal_baseline(b_traj_cp)
    # scalar, ave_val = cal_scalar_error(br_baseline, m_baseline)

    # correct the scalar
    corrected_traj = scale_corrector(m_traj, b_traj_cp_noise)
    corrected_baseline = blendertraj.cal_baseline(corrected_traj)
    scalar, ave_val = cal_scalar_error(br_baseline, corrected_baseline)

    # calculate RMSE
    b_m_RMSE, b_m_normlist = cal_Traj_RMSE(corrected_traj, b_traj_cp_noise)

    if PLOT_TRAJ:
        # plot_traj(b_traj_cp, m_traj, corrected_traj)
        plot_RMSE(b_traj_cp, b_traj_cp_noise, m_traj, corrected_traj)

    if PLOT_ALL:
        plot_all(b_traj_cp, corrected_traj, br_baseline, corrected_baseline, scalar, ave_val)
    plt.show()


if __name__ == '__main__':
    USE_ORIGINAL_MAP = True
    USE_RESIDENT = True
    USE_UNI = False
    ADD_NOISE = True
    LOAD_MSGDATA = True
    PLOT_TRAJ = True
    PLOT_BASELINE = False
    PLOT_ALL = False
    main()
