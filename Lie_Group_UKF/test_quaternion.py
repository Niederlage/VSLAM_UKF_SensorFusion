from mathutils import Quaternion
import numpy as np
from enum import Enum
from Lie_Group_UKF.LG_Tool import cholupdate, Lie_Group
from scipy.linalg import pascal
from scipy.spatial.transform import Rotation
import sympy as sp

class Color(Enum):
    RED = 1
    ORANGE = 2
    YELLOW = 3
    GREEN = 4
    BLUE = 5
    PURPLE = 6


color = Color(1)

q0 = [0, 0, 0, 1]
q1 = [1, 0, 0, 0]

Q0 = Quaternion(q0)
Q1 = Quaternion(q1)
rot0 = np.array(Q0.to_matrix())
rot1 = np.array(Q1.to_matrix())

a = np.arange(10).reshape(5, 2)
b = a.copy()

ta = np.array([[0, 1, 2, 3]])
tb = ta.copy()
tc = np.kron(ta, tb.T)
# np.random.shuffle(b)
tt = zip(a, b.T)
C = tuple(np.kron(i, j[:, None]) for i, j in tt)

matrixSize = 5
# A = np.random.rand(matrixSize,matrixSize)
# B = np.dot(A,A.transpose())
# C = B+B.T # makesure symmetric
# eig = np.linalg.eigvals(C)
# ca = np.linalg.cholesky(C)
# print(ca)

# x = np.array([0, 0, 0, 1 / np.sqrt(2)])[:, None]
# A = pascal(4)
# R = np.linalg.cholesky(A)
# # A_aug = np.kron(np.eye(2), A)
# # R_aug = np.linalg.cholesky(A_aug)
# # R1 = cholupdate(R.T, x, "-")
# # print(R_aug[4:8])
# # print(R_aug[4:])
# A = pascal(5)
# X = np.linalg.qr(A, mode='r')
# lg = Lie_Group()
# a = np.arange(1, 4)[:, None] * 8
# M = a @ a.T
# q, r = np.linalg.qr(M)
# theta = lg.logSO3(q)
# # print(q)
# # print(r)
# # print("det:",np.linalg.det(r))
# # print(np.linalg.norm(q))
# ac = np.zeros((3, 5))
# ac[:3, :3] = np.eye(3)
# bc = np.array([0, 0, 0, 0, 1])[:, None]
#
# test_theta = Rotation.from_quat(np.array([1, 0, 0, 0]))
# test_theta = test_theta.as_euler('zyx', degrees=True)
# print(408%20)


wx, ax, I ,dt = sp.symbols("wx ax I dt")
A = sp.Matrix([[-wx,0,0],[-ax,-wx,0],[0,I,-wx]])
expr = A *dt + A**2/2 *dt**2
print(expr)