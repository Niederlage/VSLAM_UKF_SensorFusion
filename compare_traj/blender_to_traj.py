import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import namedtuple
from mathutils import Quaternion


# header = ['Timestamp','Trans_x', 'Trans_y', 'Trans_z',
#           'Quaternion_w', 'Quaternion_x', 'Quaternion_y', 'Quaternion_z']

class Blender_to_Traj:
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

        pose = np.array(camPose_list).reshape(-1, 8)
        if 'noisy' in self.pathin:
            pose[:, 2] = -pose[:, 2]

        return pose

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
        dis_seq = []
        for i in range(len(diff_list)):
            dis_seq.append(np.linalg.norm(diff_list[i]))
            # print('d{order} = {num:.2f}m'.format(order=i, num=dis_list[i-1]))
        return np.array(dis_seq)

    def rot_traj(self, traj, theta):
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rot_traj = rot @ traj.T
        return rot_traj.T

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

            ax.plot(pose_w[0, 0], pose_w[0, 1], "X",color="red")
            ax.plot(pose_w[10, 0], pose_w[10, 1], "X",color="green")
            ax.plot(pose_w[-1, 0], pose_w[-1, 1],"X" ,color="purple")

        ax.legend()  # 画一条空间曲线
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.set_xlim(-5, 12)
        # ax.set_ylim(-6, 12)
        # ax.set_zlim(-6, 6)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


if __name__ == "__main__":
    folderpath = 'saved_data/from_blender_csv/'
    path = 'Camera_Resident_ideal_blender.csv'
    path2 = 'Camera_Uni_ideal_blender.csv'
    path3 = 'Camera_Resident_noisy_blender.csv'
    path4 = 'Camera_Uni_noisy_blender.csv'
    PLOT_TRAJ = True
    PLOT_BASELINE = False
    # visualize_pose_baseline(path)
    # visualize_pose_baseline(path)
    # visualize_pose_baseline(path2)
    b2t = Blender_to_Traj(folderpath + path)
    pose = b2t.csv_to_pose()
    real_traj = pose[:, 1:3]
    try_rot_traj = b2t.rot_traj(real_traj, -np.pi * 2/4)

    if PLOT_TRAJ:
        b2t.plot_pose(try_rot_traj)
    if PLOT_BASELINE:
        total_frame = len(pose)
        real_baseline = b2t.cal_baseline(real_traj)
        traj_name = path.replace('_blender.csv', '')
        real_traj_length = real_baseline.sum()
        print(traj_name + ' length =', real_traj_length)
        print('average driving speed = {:.3f} m/frame'.format(real_traj_length / total_frame))
        b2t.plot_baseline(real_baseline)

