from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from mathutils import Quaternion
import msgpack
import numpy as np
import os


class Msg_to_Traj:
    def __init__(self, pathin):
        self.path = pathin

    def get_trajectory(self, pose_cw):
        trans_cw = pose_cw[:, :3]
        quat_cw = pose_cw[:, 3:]
        pose_w = []
        for i, _ in enumerate(pose_cw):
            q = Quaternion(quat_cw[i])
            rot = np.array(q.to_matrix())
            pose_w.append(rot @ trans_cw[i])
        return np.array(pose_w)

    def cal_baseline(self, traj):
        diff_list = np.diff(traj, axis=0)
        baseline = []
        for i in range(len(diff_list)):
            baseline.append(np.linalg.norm(diff_list[i]))
        return np.array(baseline)

    def msg_unpack_to_array(self):
        landmark_list = []
        keyframe_scale = []
        keyframe_pose_cw = np.zeros((1, 9))
        keyframe_undists = []

        with open(self.path, "rb") as f:
            data_load = msgpack.unpackb(f.read(), use_list=False, raw=False)
            landmarks = data_load['landmarks']
            keyframes = data_load['keyframes']
            for i in landmarks:
                landmark_list.append(landmarks[i]['pos_w'])
            for j in keyframes:
                keyframe_undists.append(keyframes[j]['undists'])
                keyframe_scale.append(keyframes[j]['scale_factor'])
                src_frame_id = keyframes[j]['src_frm_id']
                kf_temp = np.hstack(([int(j), src_frame_id], np.array(keyframes[j]['trans_cw'])))
                kf_temp = np.hstack((kf_temp, np.array(keyframes[j]['rot_cw'])))
                keyframe_pose_cw = np.vstack((keyframe_pose_cw, kf_temp))
        keyframe_pose_cw = keyframe_pose_cw[keyframe_pose_cw[:, 0].argsort()]
        keyframe_pose_cw = np.delete(keyframe_pose_cw, 0, axis=0)

        return np.array(landmark_list), np.array(keyframe_scale), keyframe_pose_cw, keyframe_undists

    def plot_landmarks(self, line, display3D=True):
        fig = plt.figure()
        if display3D:
            ax = fig.gca(projection='3d')
            # # Plot the surface.
            # surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,
            #                    linewidth=0, antialiased=False,alpha = 0.5)
            ax.scatter(line[:, 0], line[:, 2], line[:, 1], label='Feather points', color='r', marker='o')
            ax.legend()  # 画一条空间曲线
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.set_xlim(-5, 12)
            ax.set_ylim(-6, 12)
            ax.set_zlim(-6, 6)

        else:
            plt.figure('Scatter Feather points')
            ax = plt.gca()
            ax.scatter(line[:, 0], line[:, 2], label='Feather points', color='r', marker='o', alpha=0.5)
            ax.legend()  # 画一条空间曲线
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            # ax.set_xlim(-6.5, 18)
            # ax.set_ylim(-3, 14)

        plt.show()

    def plot_unidists(self, unidists):
        plt.figure('Scatter Feather points')
        ax = plt.gca()
        ax.scatter(unidists[:, 0], unidists[:, 1], label='Feather points in one frame', color='b', marker='o',
                   alpha=0.5)
        ax.legend()  # 画一条空间曲线
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.set_xlim(-6.5, 18)
        # ax.set_ylim(-3, 14)
        plt.show()

    def plot_animation(self, keyframe_undists):
        for unidists in keyframe_undists:
            plt.cla()
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            ax = plt.gca()
            # for stopping simulation with the esc key.
            unidists = np.array(unidists)
            # point_num = len(unidists)
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            # for i in range(point_num):
            ax.scatter(unidists[:, 0], unidists[:, 1], label='Feather points in one frame', color='purple', marker='o',
                       alpha=0.5)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim(0, 1280)
            ax.set_ylim(0, 720)
            ax.axis("equal")
            plt.grid(True)
            plt.pause(0.1)

        plt.show()

    def plot_pose(self, pose_w, display3D=False):

        fig = plt.figure()

        if display3D:
            ax = fig.gca(projection='3d')
            ax.plot(pose_w[:, 0], pose_w[:, 1], pose_w[:, 2], "o-", label='pose points')
            ax.scatter(pose_w[0, 0], pose_w[0, 1], pose_w[0, 2], color="red")
            ax.scatter(pose_w[10, 0], pose_w[10, 1], pose_w[10, 2], color="green")
            ax.scatter(pose_w[-1, 0], pose_w[-1, 1], pose_w[-1, 2], color="purple")
            ax.set_zlabel('Z')
        else:
            timeline = np.arange(len(pose_w))

            ax = fig.add_subplot(121)
            ax.plot(pose_w[:, 0], pose_w[:, 2], "o-", label='pose points')
            ax.plot(pose_w[0, 0], pose_w[0, 2], "X", color="red")
            ax.plot(pose_w[10, 0], pose_w[10, 2], "X", color="green")
            ax.plot(pose_w[-1, 0], pose_w[-1, 2], "X", color="purple")

            ax2 = fig.add_subplot(122)
            ax2.plot(timeline, pose_w[:, 1], "o-", label='pose points')
            ax2.plot(timeline[0], pose_w[0, 1], "X", color="red")
            ax2.plot(timeline[10], pose_w[10, 1], "X", color="green")
            ax2.plot(timeline[-1], pose_w[-1, 1], "X", color="purple")

        ax.legend()  # 画一条空间曲线
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.set_xlim(-5, 12)
        # ax.set_ylim(-6, 12)
        # ax.set_zlim(-6, 6)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def plot_baseline(self, pose_traj):
        baseline = self.cal_baseline(pose_traj)

        fig = plt.gcf()
        ax1 = fig.add_subplot(121)
        ax1.plot(baseline, color='tab:red', label='ground truth')
        ax1.set_xlabel('frame', fontsize=20)
        ax1.set_ylabel('baseline', fontsize=20)
        ax1.set_title('compared OpenVSLAM and Ground Truth', fontsize=20)
        ax1.tick_params(axis='x', rotation=0, labelsize=12)
        ax1.tick_params(axis='y', rotation=0)
        ax1.grid(alpha=.4)
        ax1.grid(True)

        # # # Plot Line2 (Right Y Axis)
        ax2 = fig.add_subplot(122)
        counts, bins = np.histogram(baseline, bins=50)
        edges = np.delete(bins, 0)
        res = edges.dot(counts) / len(baseline)

        ax2.hist(bins[:-1], bins, weights=counts, facecolor='tab:red', alpha=0.75)
        print('average baseline:={:.3f}m'.format(res))
        ax2.grid(alpha=.4)
        ax2.grid(True)

        # # # ax2 (right Y axis)
        ax2.set_ylabel("frequency", color='tab:blue', fontsize=20)
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.set_xlabel('baseline/m', fontsize=20)
        ax2.set_xlim(0, 1)
        # plt.ylim(0, 1)
        ax2.set_title("Distribution of Baseline", fontsize=22)
        # plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    SAVE_MSG = True
    PLOT_POSE = True
    LOAD_DATA = False
    PLOT_LANDMARKS = False
    PLOT_LANDMARKS_3D = False
    PLOT_ANIMATION = False

    path0 = "/home/taungdrier/Desktop/FAUbox/Archiv_OpenVSLAM/VSLAM_results/eval_uni_EQT_2400_15102020/"
    # path = 'test_simu_equirectangular_resident.msg'
    # path = 'resident_EQT_0p5.msg'
    # path = 'resident_EQT_0p5_skip0_kp2000_noloop.msg'
    # path = 'neighbor_PPT.msg'
    # path = 'uni_EQT_1200_1p05.msg'
    # filename = "uni_EQT_2400_skip0_kp2000_noloop_s1p2.msg"

    filename = "eval_uni_EQT_2400_15102020.msg"
    folderpath = 'saved_data/from_msg_data/'
    path = path0 + filename
    m2t = Msg_to_Traj(path)
    saved_name = "saved_" + filename.replace(".msg", "")

    if SAVE_MSG:
        landmarks, keyframe_scale, keyframe_pose_cw, keyframe_undists = m2t.msg_unpack_to_array()
        try:
            os.mkdir(folderpath)

        except:
            print('folder already exist ...')
        np.savez_compressed(folderpath + saved_name, landmarks=landmarks, keyframe_scale=keyframe_scale,
                            keyframe_pose_cw=keyframe_pose_cw)
        # np.save(folderpath + 'landmarks.npy', landmarks)
        # np.save(folderpath + 'keyframe_scale.npy', keyframe_scale)
        # np.save(folderpath + 'keyframe_pose_cw.npy', keyframe_pose_cw)
        # np.save(folderpath + 'keyframe_unidists.npy', keyframe_undists[6])
        print('file saved successfully...')
        if PLOT_ANIMATION:
            m2t.plot_animation(keyframe_undists)
    else:
        LOAD_DATA = True

    if LOAD_DATA:
        load_data = np.load(folderpath + 'saved_uni_EQT_2400_skip0_kp2000_noloop_s1p2.npz')
        print('file loaded successfully...')
        kf_pose_cw = load_data['keyframe_pose_cw']
        lm = load_data['landmarks']
    else:
        kf_pose_cw = keyframe_pose_cw
        lm = landmarks

    if PLOT_POSE:
        pose_cw = m2t.get_trajectory(kf_pose_cw[:, 2:])
        m2t.plot_pose(pose_cw)
        m2t.plot_baseline(pose_cw)

    if PLOT_LANDMARKS:

        if PLOT_LANDMARKS_3D:
            m2t.plot_landmarks(lm)
            print("landmark plot successfully...")
        else:
            m2t.plot_landmarks(lm, display3D=False)
