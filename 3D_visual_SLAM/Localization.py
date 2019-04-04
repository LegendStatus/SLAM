from utils import *
import os
import cv2
from model import *
import numpy as np
from scipy.linalg import expm, logm, cosm, sinm, inv

dataset = 20
t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(os.getcwd() + '\\data\\00%d.npz'%dataset)

'''IMU localization'''
# Preprocessing
M = np.zeros((4, 4))
M[:, 0:3] = np.tile(K[0:2, :], (2, 1))
M[2, 3] = -b*K[0, 0]  # Intrinsic Calibration
tau = np.zeros((t.shape[1]-1, 1))
for i in range(t.shape[1]-1):
    tau[i] = t[0, i+1] - t[0, i]  # time stamps
mu = np.zeros((4, 4, t.shape[1]))
mu[:, :, 0] = np.eye(4)  # mean initialization
cov = np.zeros((6, 6, t.shape[1]))
cov[:, :, 0] = np.eye(6)  # initial covariance
w_T_imu = np.zeros((4, 4, t.shape[1]))
w_T_imu[:, :, 0] = inv(mu[:, :, 0])  # world to imu frame transformation initialization


# Initialization
for i in range(len(tau)):
    noise_motion = 1000*np.eye(6)  # covariance of noise in motion model
    vt = linear_velocity[:, i+1]
    wt = rotational_velocity[:, i+1]
    twist_control = np.zeros((4, 4))
    twist_control[0:3, 0:3] = hat_map(wt)
    twist_control[0:3, 3] = vt.T
    mu[:, :, i+1] = expm(-tau[i]*twist_control).dot(mu[:, :, i])
    w_T_imu[:, :, i + 1] = inv(mu[:, :, i + 1])
    control_se3 = np.zeros((6, 6))
    control_se3[0:3, 0:3] = hat_map(wt)
    control_se3[3:6, 3:6] = hat_map(wt)
    control_se3[0:3, 3:6] = hat_map(vt)
    cov[:, :, i+1] = expm(-tau[i]*control_se3).dot(cov[:, :, i].dot(expm(-tau[i]*control_se3).T)) + tau[i]**2*noise_motion
    # print('current iteration: %d' % i)







def visualize(pose,path_name="Unknown",show_ori=False):
  '''
  function to visualize the trajectory in 2D
  Input:
      pose:   4*4*N matrix representing the camera pose,
              where N is the number of pose, and each
              4*4 matrix is in SE(3)
  '''
  fig,ax = plt.subplots(figsize=(5,5))
  n_pose = pose.shape[2]
  ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
  ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
  ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  # ax.scatter(x,y,c='g',linewidth=0.1,label='landmark')
  if show_ori:
      select_ori_index = list(range(0,n_pose,int(n_pose/50)))
      yaw_list = []
      for i in select_ori_index:
          _,_,yaw = mat2euler(pose[:3,:3,i])
          yaw_list.append(yaw)
      dx = np.cos(yaw_list)
      dy = np.sin(yaw_list)
      dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
      ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
          color="b",units="xy",width=1)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.axis('equal')
  ax.grid(False)
  ax.legend()
  plt.show(block=True)
  return fig, ax





visualize(w_T_imu, path_name="Dead-Reckoning", show_ori=True)
np.savez('trajectory_%d.npz' % dataset, mu=mu, w_T_imu=w_T_imu)




