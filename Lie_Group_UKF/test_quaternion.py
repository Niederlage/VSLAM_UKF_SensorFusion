from mathutils import Quaternion
import numpy as np

q0 = [0,0,0,1]
q1 = [1,0,0,0]

Q0 = Quaternion(q0)
Q1 = Quaternion(q1)
rot0 = np.array(Q0.to_matrix())
rot1 = np.array(Q1.to_matrix())

print(1)