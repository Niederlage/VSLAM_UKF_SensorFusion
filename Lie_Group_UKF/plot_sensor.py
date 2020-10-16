from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from mathutils import Quaternion
import csv
from collections import namedtuple
import numpy as np
from compare_traj.msg_to_traj import Msg_to_Traj
from compare_traj.blender_to_traj import Blender_to_Traj
from scipy.spatial.transform import Rotation as R


class VSLAM_to_Traj:
    def __init__(self, pathin):
        self.pathin = pathin

    def csv_to_pose(self):
        camPose_list = []
        with open(self.pathin) as cam_csv:
            camdata = csv.reader(cam_csv)
            headers = next(camdata)
            Row = namedtuple('Row', headers)
            for r in camdata:
                row = Row(*r)
                for temp in row:
                    camPose_list.append(float(temp))

        pose = np.array(camPose_list).reshape(-1, 9)
        if 'noisy' in self.pathin:
            pose[:, 2] = -pose[:, 2]

        return pose

    def get_trajectory(self, pose_wc):
        trans_wc = pose_wc[:, :3]
        quat_wc = pose_wc[:, 3:]
        # quat_wc[:,[0,1,2,3]] = quat_wc[:,[3,0,1,2]]
        pose_w = []
        for i, _ in enumerate(pose_wc):
            q = Quaternion(quat_wc[i])
            rot = np.array(q.to_matrix())
            # q = R.from_quat(quat_wc[i])
            # rot = np.array(R.as_matrix(q))
            # pose_w.append(rot)
            pose_w.append(rot @ trans_wc[i])
        return np.array(pose_w)

    def cal_baseline(self, traj):
        diff_list = np.diff(traj, axis=0)
        dis_seq = []
        for i in range(len(diff_list)):
            dis_seq.append(np.linalg.norm(diff_list[i]))
            # print('d{order} = {num:.2f}m'.format(order=i, num=dis_list[i-1]))
        return np.array(dis_seq)

    def cal_velocity(self,traj):
        diff_list = np.diff(traj, axis=0)
        return diff_list * 30

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

    def plot_baseline(self, baseline):
        # baseline = np.delete(baseline, 0)

        fig = plt.figure()
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
        counts, bins = np.histogram(baseline)
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
        # plt.ylim(0, 1)
        ax2.set_title("Distribution of Baseline", fontsize=22)
        # plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # path0 = "/home/taungdrier/Desktop/FAUbox/Archiv_OpenVSLAM/VSLAM_results/eval_uni_EQT_2400_15102020/"
    # filename = "eval_uni_EQT_2400_15102020.msg"
    folderpath = '../compare_traj/saved_data/from_msg_data/'
    # posecsv = "uni_EQT_traj.csv"
    posecsv = "uni_EQT_kf_traj.csv"
    fullposecsv = "frame_trajectory.csv"
    # m2t = Msg_to_Traj(path0 + filename)
    # load_data = np.load(folderpath + 'saved_eval_uni_EQT_2400_15102020.npz')
    # kf_pose_cw = load_data['keyframe_pose_cw']
    # zw = kf_pose_cw[kf_pose_cw[:, 0].argsort()]  # 按第'1'列排序
    # pose_cw = m2t.get_trajectory(zw[:, 2:])
    # v2t = VSLAM_to_Traj(path0 + posecsv)


    v2t = VSLAM_to_Traj(fullposecsv)
    full_pose = v2t.csv_to_pose()
    real_traj = v2t.get_trajectory(full_pose[:, 2:])
    # to regular view transformation
    real_traj[:, [1, 2]] = real_traj[:, [2, 1]]
    real_traj[:, 1] = -real_traj[:, 1]
    full_baseline = v2t.cal_baseline(real_traj)

    # PLOT
    v2t.plot_pose(real_traj, display3D=True)
    # v_traj = v2t.cal_velocity(real_traj)
    # v2t.plot_pose(v_traj, display3D=True)
    # v2t.plot_baseline(30 *full_baseline)
