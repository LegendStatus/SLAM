import numpy as np


def pi(x):
    return x/x[2]


def rotation(theta):
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(theta)
    R[1, 1] = np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta)
    R[2, 2] = 1
    return R


def imu2world(state):
    w_T_i = np.zeros((4, 4))
    w_T_i[0:3, 0:3] = rotation(state[3])
    w_T_i[0, 3] = state[0]
    w_T_i[1, 3] = state[1]
    w_T_i[2, 3] = state[2]
    return w_T_i


def der_pi(x):
    der = np.zeros((4, 4))
    der[0, 0] = 1
    der[1, 1] = 1
    der[3, 3] = 1
    der[0, 2] = -x[0]/x[2]
    der[1, 2] = -x[1]/x[2]
    der[3, 2] = -x[3]/x[2]
    der = der * (1/x[2])
    return der


def hat_map(x):
    hat = np.zeros((3, 3))
    hat[2, 1] = x[0]
    hat[1, 2] = -x[0]
    hat[0, 2] = x[1]
    hat[2, 0] = -x[1]
    hat[1, 0] = x[2]
    hat[0, 1] = -x[2]
    return hat


def inv_hat_map(x):
    inv_hat = np.zeros((3, 1))
    inv_hat[0] = x[2, 1]
    inv_hat[1] = x[0, 2]
    inv_hat[2] = x[1, 0]
    return inv_hat


def SE3tostate(mu):
    # return
    ang = inv_hat_map()


def dot_map(x):
    dot = np.zeros((4, 6))
    dot[0:3, 0:3] = x[3]*np.eye(3)
    dot[0:3, 3:6] = -hat_map(x[0:3])
    return dot



