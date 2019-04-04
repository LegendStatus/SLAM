from utils import *
import os
import cv2
from model import *
import numpy as np
from scipy.linalg import expm, logm, cosm, sinm, inv

dataset = 20
t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(os.getcwd()+'\\data\\00%d.npz'%dataset)

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
mu = np.zeros((4, 4, t.shape[1]))
mu[:, :, 0] = np.eye(4)  # mean initialization
# pos_landmark = np.zeros((4, features.shape[1], t.shape[1]))
# pos_landmark[3, :, :] = 1  # homogeneous coordinates of landmarks in world frame
pos_landmark = np.zeros((4, features.shape[1]))
pos_landmark[3, :] = 1  # homogeneous coordinates of landmarks in world frame
cov = np.zeros((6, 6, t.shape[1]))
cov[:, :, 0] = np.eye(6)  # initial covariance
# cov_landmark = np.zeros((3*features.shape[1], 3*features.shape[1], t.shape[1]))
# cov_landmark[:, :, 0] = 0.1*np.eye(3*features.shape[1])  # initial corresponding covariance
# cov_landmark = np.zeros((3*features.shape[1], 3*features.shape[1]))
cov_landmark = np.eye(3*features.shape[1])

w_T_imu = np.zeros((4, 4, t.shape[1]))
w_T_imu[:, :, 0] = inv(mu[:, :, 0])  # world to imu frame transformation initialization
landmark = []
obsv_landmark = []

w_T_imu_temp = np.zeros((4, 4, t.shape[1]))
w_T_imu_temp[:, :, 0] = inv(mu[:, :, 0])

# visual SLAM
for i in range(len(tau)):

    # EKF predict
    noise_motion = 0.00001 * np.eye(6)  # covariance of noise in motion model
    vt = linear_velocity[:, i + 1]
    wt = rotational_velocity[:, i + 1]
    twist_control = np.zeros((4, 4))
    twist_control[0:3, 0:3] = hat_map(wt)
    twist_control[0:3, 3] = vt.T
    mu[:, :, i] = expm(-tau[i] * twist_control).dot(mu[:, :, i])
    w_T_imu[:, :, i] = inv(mu[:, :, i])
    w_T_imu_temp[:, :, i] = inv(mu[:, :, i])
    control_se3 = np.zeros((6, 6))
    control_se3[0:3, 0:3] = hat_map(wt)
    control_se3[3:6, 3:6] = hat_map(wt)
    control_se3[0:3, 3:6] = hat_map(vt)
    cov[:, :, i] = expm(-tau[i] * control_se3).dot(cov[:, :, i].dot(expm(-tau[i] * control_se3).T)) + tau[
        i] ** 2 * noise_motion

    # Back projection initialization of landmarks
    scan_ind = np.unique(np.where(features[:, :, i] != -1)[1])
    # Only back-project landmarks on those observed for first time
    scan_none = np.unique(np.where(pos_landmark == 0)[1])
    scan_intersect = [val for val in scan_ind if val in scan_none]
    scan_ind = scan_intersect
    # Inverse stereo-camera model
    disparity = features[0, scan_ind, i] - features[2, scan_ind, i]
    z = K[0, 0] * b / disparity
    x = z * (features[0, scan_ind, i] - K[0, 2]) / K[0, 0]
    y = z * (features[1, scan_ind, i] - K[1, 2]) / K[1, 1]
    state = np.vstack((x, y, z))
    pos_landmark[0:3, scan_ind] = np.linalg.pinv(w_T_imu[0:3, 0:3, i].T).dot(np.linalg.inv(Roc).dot(state)) + \
                                  np.tile(np.expand_dims(w_T_imu[0:3, 3, i], axis=1), [1, len(scan_ind)])

    # Keep track on all observed landmarks for time t
    scan_ind = np.unique(np.where(features[:, :, i] != -1)[1])
    obsv_landmark.append(scan_ind)

    # EKF update
    if scan_ind.size == 0:
        mu[:, :, i + 1] = mu[:, :, i]
        # pos_landmark[:, :, i+1] = pos_landmark[:, :, i]
        cov[:, :, i + 1] = cov[:, :, i]
    else:
        noise_obsv = 50000000 * np.eye(4*len(scan_ind))  # covariance of noise in observation model
        # noise_obsv = 0.00001*np.random.random((4*len(scan_ind), 4*len(scan_ind)))
        H = np.zeros((4 * len(scan_ind), 6))
        predict_scan = np.zeros((4, len(scan_ind)))
        H_landmark = np.zeros((4 * len(scan_ind), 3 * features.shape[1]))
        D_landmark = np.zeros((4 * features.shape[1], 3 * features.shape[1]))
        for j in range(len(scan_ind)):
            H[4 * j:4 * (j + 1), :] = M.dot(der_pi(cam_T_imu.dot(mu[:, :, i]).dot(pos_landmark[:, scan_ind[j]]))
                                            .dot(cam_T_imu).dot(dot_map(mu[:, :, i].dot(pos_landmark[:, scan_ind[j]]))))
            predict_scan[:, j] = M.dot(pi(cam_T_imu.dot(mu[:, :, i].dot(pos_landmark[:, scan_ind[j]]))))
            H_landmark[4 * j:4 * (j + 1), 3 * scan_ind[j]:3 * (scan_ind[j] + 1)] = M.dot(
                der_pi(cam_T_imu.dot(mu[:, :, i].dot(pos_landmark[:, scan_ind[j]])))
                    .dot(cam_T_imu.dot(mu[:, :, i].dot(dilation))))
            D_landmark[4 * scan_ind[j]:4 * (scan_ind[j] + 1), 3 * scan_ind[j]:3 * (scan_ind[j] + 1)] = dilation
        obsv_scan = features[:, scan_ind, i]
        scan_diff = (obsv_scan - predict_scan).reshape(4 * len(scan_ind), 1, order='F')
        kalman_gain = cov[:, :, i].dot(H.T).dot(inv(H.dot(cov[:, :, i].dot(H.T)) + noise_obsv))
        twist = np.zeros((4, 4))
        twist[0:3, 0:3] = hat_map(kalman_gain.dot(scan_diff)[3:6])
        twist[0:3, 3] = kalman_gain.dot(scan_diff)[0:3].T
        mu[:, :, i + 1] = expm(twist).dot(mu[:, :, i])
        w_T_imu[:, :, i + 1] = inv(mu[:, :, i + 1])  # imu in world frame pose
        cov[:, :, i + 1] = (np.eye(6) - kalman_gain.dot(H)).dot(cov[:, :, i])
        # update landmark position
        kalman_gain_landmark = cov_landmark.dot(H_landmark.T).dot(np.linalg.pinv(H_landmark.dot(cov_landmark.dot(H_landmark.T)) + noise_obsv))
        pos_landmark = pos_landmark + D_landmark.dot(kalman_gain_landmark.dot(scan_diff)).reshape(4, features.shape[1], order='F')
        cov_landmark = (np.eye(3 * features.shape[1]) - kalman_gain_landmark.dot(H_landmark)).dot(cov_landmark)
    # landmark.append(pos_landmark[:, scan_ind, i + 1])
    print(pos_landmark[:, scan_ind])
    print('Current iteration: %d' % (i + 1))


visualize_trajectory_2d(pos_landmark[0,:], pos_landmark[1,:], w_T_imu, path_name="visualSLAM", show_ori=True)
x=95; visualize_trajectory_2d(pos_landmark[0, obsv_landmark[x], x], pos_landmark[1, obsv_landmark[x], x], w_T_imu[:, :, 0:x], path_name='Unknown', show_ori=False)



