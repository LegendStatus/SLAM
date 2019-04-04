import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
import os


dataset = 27


def visualize(x1,y1,x2,y2,pose,path_name="Unknown",show_ori=False):
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
  ax.scatter(x1, y1,c='b',linewidth=0.1,label='feature_unique')
  ax.scatter(x2, y2, c='r', linewidth=0.1, label='feature_nonunique')
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


with np.load(os.getcwd()+'\\Param\\feature_same_unique_obsv_%d.npz' % dataset) as data:
    mu1 = data['mu']
with np.load(os.getcwd()+'\\Param\\feature_same_nonunique_obsv_%d.npz' % dataset) as data:
    mu2 = data['mu']
with np.load('trajectory_%d.npz' % dataset) as data:
    imu_T_w = data['mu']
    w_T_imu = data['w_T_imu']

visualize(mu1[0,:,-1], mu1[1,:,-1], mu2[0,:,-1], mu2[1,:,-1], w_T_imu, path_name='Comparasion', show_ori=False)
