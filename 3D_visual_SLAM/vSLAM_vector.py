from utils import *
import os
import cv2
from model import *
import numpy as np
from scipy.linalg import expm, logm, cosm, sinm, inv

dataset = 42
t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(os.getcwd()+'/data/00%d.npz'%dataset)

# Preprocessing
Roc = np.zeros((3, 3))
Roc[0, 1] = -1
Roc[1, 2] = -1
Roc[2, 0] = 1  # rotation from regular to optical frame
M = np.zeros((4, 4))
M[:, 0:3] = np.tile(K[0:2, :], (2, 1))
M[2, 3] = -b*K[0, 0]  # Intrinsic Calibration
tau = np.zeros((t.shape[1]-1, 1))
for i in range(t.shape[1]-1):
    tau[i] = t[0, i+1] - t[0, i]  # time stamps
dilation = np.vstack((np.eye(3), np.array([0, 0, 0])))  # dilation matrix

# Initialization
mu = np.zeros((4, 4+features.shape[1], t.shape[1]))
mu[:, 0:4, 0] = np.eye(4)  # mean initialization
mu[3, 4:4+features.shape[1], :] = 1
cov = np.zeros((6+3*features.shape[1], 6+3*features.shape[1], t.shape[1]))
cov[:, :, 0] = 0.5 * np.eye(6+3*features.shape[1])  # initial covariance

w_T_imu = np.zeros((4, 4, t.shape[1]))
w_T_imu[:, :, 0] = inv(mu[:, 0:4, 0])  # world to imu frame transformation initialization

# visual SLAM
for i in range(len(tau)):

    # EKF predict
    noise_motion = 0.1 * np.eye(6)  # covariance of noise in motion model
    vt = linear_velocity[:, i + 1]
    wt = rotational_velocity[:, i + 1]
    twist_control = np.zeros((4, 4))
    twist_control[0:3, 0:3] = hat_map(wt)
    twist_control[0:3, 3] = vt.T
    mu[:, 0:4, i] = expm(-tau[i] * twist_control).dot(mu[:, 0:4, i])
    w_T_imu[:, :, i] = inv(mu[:, 0:4, i])
    control_se3 = np.zeros((6, 6))
    control_se3[0:3, 0:3] = hat_map(wt)
    control_se3[3:6, 3:6] = hat_map(wt)
    control_se3[0:3, 3:6] = hat_map(vt)
    cov[0:6, 0:6, i] = expm(-tau[i] * control_se3).dot(cov[0:6, 0:6, i].dot(expm(-tau[i] * control_se3).T)) + tau[
        i] ** 2 * noise_motion

    # Back projection initialization of landmarks
    scan_ind = np.unique(np.where(features[:, :, i] != -1)[1]) + 4
    # Only back-project landmarks on those observed for first time
    scan_none = np.unique(np.where(mu[:, 4:4+features.shape[1], i] == 0)[1])
    scan_intersect = [val-4 for val in scan_ind if val in scan_none]
    scan2 = [val for val in scan_ind if val in scan_none]
    # Inverse stereo-camera model
    disparity = features[0, scan_intersect, i] - features[2, scan_intersect, i]
    z = K[0, 0] * b / disparity
    x = z * (features[0, scan_intersect, i] - K[0, 2]) / K[0, 0]
    y = z * (features[1, scan_intersect, i] - K[1, 2]) / K[1, 1]
    state = np.vstack((x, y, z))
    mu[0:3, scan2, i] = np.linalg.pinv(w_T_imu[0:3, 0:3, i].T).dot(np.linalg.inv(Roc).dot(state)) + \
                                  np.tile(np.expand_dims(w_T_imu[0:3, 3, i], axis=1), [1, len(scan2)])

    # Keep track on all observed landmarks for time t
    # scan_ind = np.unique(np.where(features[:, :, i] != -1)[1]) + 4
    # obsv_landmark.append(scan_ind)

    # EKF update
    noise_obsv = 2000000000 * np.eye(4 * len(scan_ind))  # covariance of noise in observation model
    if scan_ind.size == 0:
        mu[:, :, i + 1] = mu[:, :, i]
        # pos_landmark[:, :, i+1] = pos_landmark[:, :, i]
        cov[:, :, i + 1] = cov[:, :, i]
    else:
        H = np.zeros((4 * len(scan_ind), 6+3*features.shape[1]))
        D = np.zeros((4*features.shape[1], 3*features.shape[1]))
        predict_scan = np.zeros((4, len(scan_ind)))
        for j in range(len(scan_ind)):
            H[4 * j:4 * (j + 1), 0:6] = M.dot(der_pi(cam_T_imu.dot(mu[:, 0:4, i]).dot(mu[:, scan_ind[j], i]))
                                            .dot(cam_T_imu).dot(dot_map(mu[:, 0:4, i].dot(mu[:, scan_ind[j], i]))))
            predict_scan[:, j] = M.dot(pi(cam_T_imu.dot(mu[:, 0:4, i].dot(mu[:, scan_ind[j], i]))))
            H[4 * j:4 * (j + 1), 3 * (scan_ind[j]-4)+6:6+3 * (scan_ind[j]-3)] = M.dot(
                der_pi(cam_T_imu.dot(mu[:, 0:4, i].dot(mu[:, scan_ind[j], i])))
                    .dot(cam_T_imu.dot(mu[:, 0:4, i].dot(dilation))))
            D[4 * (scan_ind[j]-4):4 * (scan_ind[j]-3), 3 * (scan_ind[j]-4):3 * (scan_ind[j]-3)] = dilation
        obsv_scan = features[:, scan_ind-4, i]
        scan_diff = (obsv_scan - predict_scan).reshape(4 * len(scan_ind), 1, order='F')
        kalman_gain = cov[:, :, i].dot(H.T).dot(inv(H.dot(cov[:, :, i].dot(H.T)) + noise_obsv))
        twist = np.zeros((4, 4))
        twist[0:3, 0:3] = hat_map(kalman_gain[0:6, :].dot(scan_diff)[3:6])
        twist[0:3, 3] = kalman_gain[0:6, :].dot(scan_diff)[0:3].T
        mu[:, 0:4, i + 1] = expm(twist).dot(mu[:, 0:4, i])  # mu of pose
        mu[:, 4:4+features.shape[1], i+1] = mu[:, 4:4+features.shape[1], i] + \
                                            D.dot(kalman_gain[6:6+3*features.shape[1], :].dot(scan_diff))\
                                                .reshape(4, features.shape[1], order='F')
        w_T_imu[:, :, i + 1] = inv(mu[:, 0:4, i + 1])  # imu in world frame pose
        cov[:, :, i + 1] = (np.eye(6+3*features.shape[1]) - kalman_gain.dot(H)).dot(cov[:, :, i])
    # landmark.append(pos_landmark[:, scan_ind, i + 1])
    # print(mu[:, scan_ind, i+1])
    # print('Current iteration: %d' % (i + 1))


visualize_trajectory_2d(mu[0,4:4+features.shape[1],-1], mu[1,4:4+features.shape[1],-1], w_T_imu, path_name="visualSLAM", show_ori=True)
x=95