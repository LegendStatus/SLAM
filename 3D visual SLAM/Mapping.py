from utils import *
import os
from model import *
import numpy as np
from scipy.linalg import expm, logm, cosm, sinm, inv

dataset = 42
t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(os.getcwd()+'/data/00%d.npz'%dataset)
with np.load('trajectory_%d.npz' % dataset) as data:
    imu_T_w = data['mu']
    w_T_imu = data['w_T_imu']


'''Landmark mapping'''
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
mu = np.zeros((4, features.shape[1], t.shape[1]))
mu[3, :, :] = 1  # homogeneous coordinates of landmarks in world frame
cov = np.zeros((3*features.shape[1], 3*features.shape[1], t.shape[1]))
cov[:, :, 0] = 0.1*np.eye(3*features.shape[1])  # initial corresponding covariance
obsv_landmark = []
o_T_cam = np.eye(4)
o_T_cam[0:3, 0:3] = Roc
o_T_imu = o_T_cam.dot(cam_T_imu)
landmark = []
obsv_landmark = []


# Initialization
# using back projection of stereo camera model
'''
scan_ind = np.unique(np.where(features[:, :, 0] != -1)[1])
disparity = features[0, scan_ind, 0] - features[2, scan_ind, 0]
z = K[0, 0]*b/disparity
x = z*(features[0, scan_ind, 0] - K[0, 2])/K[0, 0]
y = z*(features[1, scan_ind, 0] - K[1, 2])/K[1, 1]
# x_dot = z*(features[2, scan_ind, 0] + K[0, 0]*b - K[0, 2])/K[0, 0]
# state = np.linalg.pinv(M).dot(features[:, scan_ind, 0])[0:3]/z
state = np.vstack((x, y, z))
mu[0:3, scan_ind, 0] = np.linalg.inv(imu_T_w[0:3, 0:3, 0].T).dot(np.linalg.inv(Roc).dot(state)) + \
                       np.tile(np.expand_dims(imu_T_w[0:3, 3, 0], axis=1), [1, len(scan_ind)])
landmark = mu[:, scan_ind, 0]
obsv_landmark.append(scan_ind)
print('Current iteration : 0')
print(mu[:, scan_ind, 0])
'''

# Update process
for i in range(len(tau)):
    # back projection initialization
    scan_ind = np.unique(np.where(features[:, :, i] != -1)[1])
    # Only back-project landmarks on those has not been observed
    scan_none = np.unique(np.where(mu[:, :, i] == 0)[1])
    scan_intersect = [val for val in scan_ind if val in scan_none]
    scan_ind = scan_intersect
    # Inverse stereo-camera model
    disparity = features[0, scan_ind, i] - features[2, scan_ind, i]
    z = K[0, 0] * b / disparity
    x = z * (features[0, scan_ind, i] - K[0, 2]) / K[0, 0]
    y = z * (features[1, scan_ind, i] - K[1, 2]) / K[1, 1]
    state = np.vstack((x, y, z))
    mu[0:3, scan_ind, i] = np.linalg.inv(w_T_imu[0:3, 0:3, i].T).dot(np.linalg.inv(Roc).dot(state)) + \
                           np.tile(np.expand_dims(w_T_imu[0:3, 3, i], axis=1), [1, len(scan_ind)])

    # Keep track on all observed landmarks on
    scan_ind = np.unique(np.where(features[:, :, i] != -1)[1])
    obsv_landmark.append(scan_ind)

    # EKF update
    if len(scan_ind) == 0:
        mu[:, :, i+1] = mu[:, :, i]
        cov[:, :, i+1] = cov[:, :, i]
    else:
        noise_obsv = 15*np.eye(4*len(scan_ind))  # covariance of noise in observation model
        H = np.zeros((4 * len(scan_ind), 3 * features.shape[1]))
        D = np.zeros((4*features.shape[1], 3*features.shape[1]))
        # predict_scan = M.dot(pi(cam_T_imu.dot(imu_T_w[:, :, i].dot(mu[:, scan_ind, i]))))
        predict_scan = np.zeros((4, len(scan_ind)))
        obsv_scan = features[:, scan_ind, i]
        for j in range(len(scan_ind)):
            H[4 * j:4 * (j + 1), 3 * scan_ind[j]:3 * (scan_ind[j] + 1)] = M.dot(
                der_pi(cam_T_imu.dot(imu_T_w[:, :, i+1].dot(mu[:, scan_ind[j], i])))
                .dot(cam_T_imu.dot(imu_T_w[:, :, i+1].dot(dilation))))
            D[4*scan_ind[j]:4*(scan_ind[j] + 1), 3*scan_ind[j]:3*(scan_ind[j] + 1)] = dilation
            predict_scan[:, j] = M.dot(pi(cam_T_imu.dot(imu_T_w[:, :, i+1].dot(mu[:, scan_ind[j], i]))))
        scan_diff = (obsv_scan - predict_scan).reshape(4 * len(scan_ind), 1, order='F')
        kalman_gain = cov[:, :, i].dot(np.transpose(H).dot(inv(H.dot(cov[:, :, i].dot(H.T)) + noise_obsv)))
        mu[:, :, i + 1] = mu[:, :, i] + D.dot(kalman_gain.dot(scan_diff)).reshape(4, features.shape[1], order='F')
        cov[:, :, i + 1] = (np.eye(3 * features.shape[1]) - kalman_gain.dot(H)).dot(cov[:, :, i])
    landmark.append(mu[:, scan_ind, i+1])
    # print('Current iteration: %d' % (i + 1))
    # print(mu[:, scan_ind, i+1])


visualize_trajectory_2d(mu[0, :, -1], mu[1, :, -1], w_T_imu, path_name="Mapping", show_ori=True)
# np.savez('feature_same_unique_obsv_%d.npz'%dataset, mu=mu)
# x=180; visualize_trajectory_2d(mu[0, obsv_landmark[x], x], mu[1, obsv_landmark[x], x], w_T_imu[:, :, 0:x], path_name='Unknown', show_ori=False)



