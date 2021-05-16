import numpy as np
from scipy.linalg import expm
from skewsym import skew_w, skew_s


H_1 = 0.089
W_1 = 0.109
L_1 = 0.425
L_2 = 0.392
W_2 = 0.082
H_2 = 0.095

M = np.array([[-1,0,0,L_1+L_2],[0,0,1,W_1+W_2],[0,1,0,H_1-H_2],[0,0,0,1]])

s_1 = skew_s(np.array([[0,0,1,0,0,0]]).T)
s_2 = skew_s(np.array([[0,1,0,-H_1,0,0]]).T)
s_3 = skew_s(np.array([[0,1,0,-H_1,0,L_1]]).T)
s_4 = skew_s(np.array([[0,1,0,-H_1,0,L_1+L_2]]).T)
s_5 = skew_s(np.array([[0,0,-1,-W_1,L_1+L_2,0]]).T)
s_6 = skew_s(np.array([[0,1,0,H_2-H_1,0,L_1+L_2]]).T)

screw_axes = [s_1, s_2, s_3, s_4, s_5, s_6]

# forward kinematics using product of matrix exponentials
def fkine(angles):
    T = np.eye(4)
    for x in zip(screw_axes,angles):
        w = x[0][:3,:3]
        v = x[0][:3,3].reshape(3,1)
        angle = x[1]
        expm_w = expm(w*angle)
        g = (np.eye(3)*angle + (1-np.cos(angle))*w 
        + (angle-np.sin(angle))*np.matmul(w,w))
        gv = np.matmul(g,v)
        expm_s = np.concatenate((expm_w,gv), axis=1)
        expm_s = np.concatenate((expm_s,[[0,0,0,1]]), axis=0)
        T = np.matmul(T,expm_s)
    T = np.matmul(T,M)
    return T
    

if __name__ == '__main__':
    angles = [0,-np.pi/2,0,0,np.pi/2,0]
    T = fkine(angles)
    print(T)