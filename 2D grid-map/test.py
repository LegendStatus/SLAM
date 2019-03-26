from data import map_utils as mp
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
import os, model
import time

def tic():
  return time.time()
def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))


'''Reading data'''
os.chdir(os.getcwd()+'\\data')
dataset = 23
with np.load("Encoders%d.npz" % dataset) as data:
    encoder_counts = data["counts"]  # 4 x n encoder counts
    encoder_stamps = data["time_stamps"]  # encoder time stamps
with np.load("Hokuyo%d.npz" % dataset) as data:
    lidar_angle_min = data["angle_min"]  # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"]  # end angle of the scan [rad]
    lidar_range_min = data["range_min"]  # minimum range value [m]
    lidar_range_max = data["range_max"]  # maximum range value [m]
    lidar_ranges = data["ranges"]  # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
with np.load("Imu%d.npz" % dataset) as data:
    imu_angular_velocity = data["angular_velocity"]  # angular velocity in rad/sec
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
# with np.load("Kinect%d.npz" % dataset) as data:
#    disp_stamps = data["disparity_time_stamps"]  # acquisition times of the disparity images
#    rgb_stamps = data["rgb_time_stamps"]  # acquisition times of the rgb images

'''Preprocessing'''
# Range to Lidar frame
ang = np.arange(-135, 135.25, 0.25) * np.pi / 180.0

# Encoder to Velocity of body
distr = (encoder_counts[0, :] + encoder_counts[2, :])/2*0.0022
distl = (encoder_counts[1, :] + encoder_counts[3, :])/2*0.0022
dist = (distr+distl)/2
vel = np.float64([(dist[i])/(encoder_stamps[i+1] - encoder_stamps[i]) for i in range(len(encoder_stamps)-1)])

#plt.figure()
#plt.plot(vel)




# Imu low-pass filter for yaw rate
fs_imu = np.ceil(len(imu_stamps) / (imu_stamps[len(imu_stamps)-1] - imu_stamps[0])) # Sampling frequency of IMU
b, a = signal.butter(1, 10 / (0.5 * fs_imu), 'low') # cut-off frequency set to 8HZ
yaw_rate = signal.filtfilt(b, a, imu_angular_velocity[2, :])
#plt.figure()
#plt.plot(imu_angular_velocity[2,:])
#plt.figure()
#plt.plot(yaw_rate)




idxs = []
for i in range(encoder_stamps.shape[0]):
    checking_value = encoder_stamps[i].squeeze()
    idx = np.where(abs(imu_stamps - checking_value) == min(abs(imu_stamps - checking_value)))
    if idx[0][0] == imu_stamps.shape[0] - 1:

        idxs.append(idx[0][0] - 1)
    else:
        idxs.append(idx[0][0])

# imu_stamps_fit = (imu_stamps[idxs]+imu_stamps[idxs+1])/2
# imu_angular_velocity_fit = (imu_angular_velocity[:,idxs]+imu_angular_velocity[:,idxs+1])/2
imu_stamps_fit = imu_stamps[idxs]
yaw_rate = yaw_rate[idxs]

# Time Stamps coordination for yaw_rate and linear_velocity
# In this case all time stamps are coordinated to time stamps of liadar
#vel = model.cor(vel, encoder_stamps[0:len(encoder_stamps)-1], lidar_stamps)
#vel = np.hstack((vel, np.repeat(vel[-1], len(lidar_stamps)-len(encoder_stamps)+1)))
#yaw_rate = model.cor(yaw_rate, imu_stamps, lidar_stamps)

'''Precoding tests
# Mapping Initialization test
MAP = {}
MAP['res']   = 0.05 #meters
MAP['xmin']  = -25  #meters
MAP['ymin']  = -25
MAP['ymax']  =  25
MAP['xmax']  =  25
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
MAP['center'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / (2 * MAP['res']))) # Center cell position
# Occupancy grid information
ocy = [mp.bresenham2D(0, 0, lidar_x[i,0] * (1 / MAP['res']), lidar_y[i,0] * (1 / MAP['res']),)\
       for i in range(lidar_ranges.shape[0])] 
ocy = [np.int32(x + MAP['center']) for x in ocy] # offset to center
# Lidar frame to body frame
offset_y = np.ceil((301.83 - 330.2 / 2) / 1000 * (1 / MAP['res']))
for i in range(len(ocy)):
    ocy[i][1, :] = np.int32(ocy[i][1, :] + offset_y)
# show occupancy grid in MAP
for i in range(lidar_ranges.shape[0]):
    MAP['map'][tuple(ocy[i])] = 1
MAP['map'] = np.where(MAP['map']==1, 0, 1)
plt.imshow(MAP['map'])

'''
# Dead-Reckoning test without noise at center of gravity
stamps = encoder_stamps
pose = np.zeros((3, len(stamps)))
for i in range(len(stamps)-1):
    dt = stamps[i+1]-stamps[i]
    pose[:,i+1] = model.motion(vel[i], yaw_rate[i] , pose[:,i], dt)
#plt.figure()
#plt.scatter(pose[0,0:2000], pose[1,0:2000], c='k', linewidth=1)
plt.figure()
plt.plot(pose[0,:], pose[1,:])
plt.figure()
plt.plot(vel)
confidence = 4
MAP = model.createmap(ang, lidar_ranges[:, 0], confidence)
particle_map = [None] * len(lidar_stamps)
particle_map[0] = MAP['map']  # MAP initialization
logodds_max = 15*np.log(confidence)
logodds_min = -15*np.log(confidence)

for j in range(2000):
    rescan = model.mapupdate(pose[:, j+1], MAP, ang, lidar_ranges[:, j+1], confidence)
    MAP['map'] = MAP['map'] + rescan
    particle_map[j+1] = MAP['map']
    MAP['map'][np.where(MAP['map'] > logodds_max)] = logodds_max  # constraint the log-odds within an threshold
    MAP['map'][np.where(MAP['map'] < logodds_min)] = logodds_min
    #plt.imshow(model.log2binary(MAP['map'] + rescan))
    #plt.scatter(pose[0, 0:j+1] * 10, pose[1, 0:j+1] * 10, c='k', linewidths=1)
    #plt.pause(0.00001)
    #plt.title("current iteration is %d" % j)
    #plt.clf()

plt.figure()
plt.imshow(model.log2binary(MAP['map']))
'''
# Prediction test with 100 particles
N_particle = 50
state = [] # control input with noise
for i in range(N_particle):
    pose = np.zeros((3, len(lidar_stamps)))
    noise = np.random.randn(len(lidar_stamps), 2) / 10
    #noise = np.random.multivariate_normal(np.array([0, 0]), np.array([[0.05, 0], [0, 0.05]]), len(lidar_stamps))
    for j in range(len(lidar_stamps) - 1):
        pose[:, j+1] = model.motion(vel[j], yaw_rate[j], pose[:, j], lidar_stamps[j+1]-lidar_stamps[j])
        pose[0, j+1] += noise[j, 0]
        pose[1, j + 1] += noise[j, 1]
    state.append(pose)

plt.figure()
for i in range(N_particle):
    plt.plot(state[i][0,:], state[i][1,:])
plt.show()
x = [state[i][0,1500] for i in range(N_particle)];
y = [state[i][1,1500] for i in range(N_particle)];
plt.figure()
plt.scatter(x,y)



# Prue prediction
N_particle = 100
state = [np.zeros((3, N_particle))] * len(lidar_stamps)  # Initialization
for j in range(len(lidar_stamps)-1):
    ts = tic()
    # 2-D Gaussian noise with on control input
    noise = np.random.multivariate_normal(np.array([0, 0]), np.array([[0.1, 0], [0, 0.1]]), N_particle)
    veln = vel[j] + noise[:, 0]  # noised plane linear velocity
    yaw_raten = yaw_rate[j] + noise[:, 1]  # noised yaw rate
    pose = state[j]   # state of particles at current time stamp
    pose_t1 = np.zeros((3, N_particle))  # state of particles need to be predicted
    for i in range(N_particle):
        pose_t1[:, i] = (model.motion(veln[i], yaw_raten[i], pose[:, i], lidar_stamps[j+1]-lidar_stamps[j]))
    state[j+1] = pose_t1
    # toc(ts, "Update for one time step")



plt.figure()
trac = [None] * N_particle
for i in range(N_particle):
    trac[i] = np.array([x[:, i] for x in state]).T
    plt.plot(trac[i][0, :], trac[i][1, :])
    plt.pause(0.00000001)
plt.figure()
for i in range(len(lidar_stamps)):
    plt.scatter(trac[10][0, i], trac[10][1, i])
    plt.pause(0.05)

plt.figure(); plt.scatter(state[100][0, :], state[100][1, :]);plt.show()
plt.figure(); plt.plot(trac[50][0, 0:100], trac[50][1, 0:100]);plt.show()

'''








'''
for i in range(1000):
    plt.clf()
    MAP['map'] = MAP['map'] + update_map[i+1]
    MAP['map'][np.where(MAP['map'] > logodds_max)] = logodds_max  # constraint the log-odds within an threshold
    MAP['map'][np.where(MAP['map'] < logodds_min)] = logodds_min
    plt.imshow(model.log2binary(MAP['map']))
    plt.pause(0.000001)
    plt.title('Current iterstion: %d' % i)
    plt.scatter(tract_c[0, 0:i+1], tract_c[1, 0:i+1], c='k', linewidths=1)
'''





