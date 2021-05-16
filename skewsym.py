import numpy as np

def skew_w(mat):
    assert mat.shape == (3,1)
    ret = np.zeros((3,3))
    ret[0,1] = -mat[2,0]
    ret[0,2] = mat[1,0]
    ret[1,0] = mat[2,0]
    ret[1,2] = -mat[0,0]
    ret[2,0] = -mat[1,0]
    ret[2,1] = mat[0,0]
    return ret

def skew_s(mat):
    assert mat.shape == (6,1)
    w = skew_w(mat[:3,:])
    v = mat[3:,:]
    ret = np.concatenate((w,v), axis=1)
    ret = np.concatenate((ret,np.zeros((1,4))), axis=0)
    return ret