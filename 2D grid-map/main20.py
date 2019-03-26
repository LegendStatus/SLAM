from data import map_utils as mp
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
import os, model, sys, time
#import texture
from PIL import Image
import cv2

def tic():
  return time.time()


def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))


def view_bar(num, total):
    rate = float(num) / float(total)
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%,%d\n' % ("=" * rate_num, " " * (100 - rate_num), rate_num, num)
    sys.stdout.write(r)
    sys.stdout.flush()


'''Reading data'''
os.chdir(os.getcwd()+'\\data')
dataset = 20
with np.load("Encoders%d.npz" % dataset) as data:
    encoder_counts = data["counts"]  # 4 x n encoder counts
    encoder_stamps = data["time_stamps"]  # encoder time stamps
with np.load("Hokuyo%d.npz" % dataset) as data:
    lidar_ranges = data["ranges"]  # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
with np.load("Imu%d.npz" % dataset) as data:
    imu_angular_velocity = data["angular_velocity"]  # angular velocity in rad/sec
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
#with np.load("Kinect%d.npz" % dataset) as data:
#   disp_stamps = data["disparity_time_stamps"]  # acquisition times of the disparity images
#    rgb_stamps = data["rgb_time_stamps"]  # acquisition times of the rgb images


'''Preprocessing'''
# Range to Lidar frame
ang = np.arange(-135, 135.25, 0.25) * np.pi / 180.0

# Encoder to Velocity of body
distr = (encoder_counts[0, :] + encoder_counts[2, :])/2*0.0022
distl = (encoder_counts[1, :] + encoder_counts[3, :])/2*0.0022
dist = (distr+distl)/2
vel = np.float64([(dist[i])/(encoder_stamps[i+1] - encoder_stamps[i]) for i in range(len(encoder_stamps)-1)])


# Imu low-pass filter for yaw rate
fs_imu = np.ceil(len(imu_stamps) / (imu_stamps[len(imu_stamps)-1] - imu_stamps[0])) # Sampling frequency of IMU
b, a = signal.butter(1, 4 / (0.5 * fs_imu), 'low') # cut-off frequency set to 8HZ
yaw_rate = signal.filtfilt(b, a, imu_angular_velocity[2, :])

# Time Stamps coordination for yaw_rate and linear_velocity
# In this case all time stamps are coordinated to time stamps of liadar (encoder)
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
yaw_rate = yaw_rate[idxs]
#vel = model.cor(vel, encoder_stamps[0:len(encoder_stamps)-1], lidar_stamps)
# vel = np.hstack((vel, np.repeat(vel[-1], len(lidar_stamps)-len(encoder_stamps)+1)))
#yaw_rate = model.cor(yaw_rate, imu_stamps, lidar_stamps)


'''SLAM'''
N_particle = 100
confidence = 0.8/(1-0.8)
logodds_max = 10 * np.log(confidence)
logodds_min = -10 * np.log(confidence)

# Initialization
stamps = encoder_stamps

MAP = model.createmap(ang, lidar_ranges[:, 0], confidence)
particle_map = [None] * len(stamps)
particle_map[0] = MAP['map']  # MAP initialization
weight = np.zeros((N_particle, len(stamps)))
weight[:, 0] = 1/N_particle  # Weight initialization
state = [np.zeros((3, N_particle))] * len(stamps)
x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # x-positions of each pixel of the map
y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # y-positions of each pixel of the map
tract_c = np.zeros((2, len(stamps)))  # final tractjeory in cell grid
tract_c[:, 0] = 0
tract_xy = np.zeros((3, len(stamps))) # final tractjeory in physical xy world
tract_xy[:, 0] = 0
update_map = [None] * len(stamps)
update_map[0] = MAP['map']
# Prediction and update
tb = tic()
# correlation = [None] * len(lidar_stamps)
corr = np.zeros((N_particle, 1))  # monitor
pose_t1 = np.zeros((3, N_particle))  # state of particles need to be predicted
correction = np.zeros((2, N_particle))

tex_map = np.zeros((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8)
plt.figure(figsize=(20, 10))
for j in range(len(stamps) - 1):
    ts = tic()
    noise_x = 0.1 * np.random.normal(0, 0.5, N_particle)  # noise for v
    noise_y = 0.1 * np.random.normal(0, 0.5, N_particle)   # noise for w
    noise_yaw = 0.05 * np.random.normal(0, 0.5, N_particle)  # np.random.randn(N_particle, 1)  # noise for yaw angel
    #veln = vel[j] + noise_v  # noised plane linear velocity
    #yaw_raten = yaw_rate[j] + noise_w  # noised yaw rate
    pose = state[j]   # state of particles at current time stamp
    # Rejecting too large or too small scan
    ranges = lidar_ranges[:, j+1]
    indValid = np.logical_and((ranges < 30), (ranges > 0.1))
    ranges = ranges[indValid]
    angles = ang[indValid]
    offset_x = (301.83 - 330.2 / 2) / 1000
    for i in range(N_particle):
        pose_t1[:, i] = model.motion(vel[j], yaw_rate[j], pose[:, i], stamps[j + 1] - stamps[j])
        #pose_t1[:, i] = 0
        # body frame to world frame
        theta = (pose_t1[2, i] + noise_yaw[i])  # in radius
        x = pose_t1[0, i] + noise_x[i]
        y = pose_t1[1, i] + noise_y[i]
        xW_scan = (ranges * np.cos(angles + theta) + np.cos(theta) * offset_x + x)
        yW_scan = (ranges * np.sin(angles + theta) + np.sin(theta) * offset_x + y)
        Y = np.stack((xW_scan, yW_scan))
        # Lidar frame to body frame which is also world frame in initialization
        # 9*9 grid need to be scanned
        cent_x = (offset_x * np.cos(theta) + x)
        cent_y = (offset_x * np.sin(theta) + y)
        cent_x = np.round(x, 2) - np.round(x, 2) % MAP['res']
        cent_y = np.round(y, 2) - np.round(y, 2) % MAP['res']
        x_range = np.arange(-4*MAP['res'] + cent_x, 4*MAP['res'] + MAP['res'] + cent_x, MAP['res'])
        y_range = np.arange(-4*MAP['res'] + cent_y, 4*MAP['res'] + MAP['res'] + cent_y, MAP['res'])
        # map correlation
        c = mp.mapCorrelation(model.log2binary(MAP['map']), x_im, y_im, Y, x_range, y_range)
        corr[i] = np.max(c)
        correction[0, i] = np.where(c == np.max(c))[0][0]
        correction[1, i] = np.where(c == np.max(c))[1][0]
        #print(np.where(c == np.max(c)))
    pose_t1[2, :] = pose_t1[2, :] + noise_yaw.T
    # correlation[j] = corr
    W2resample = model.softmax(corr.T * weight[:, j])
    best_ind = np.argmax(W2resample)
    pose_best_particle = pose_t1[:, best_ind]
    # correction based on correlation
    #pose_best_particle[0] = pose_best_particle[0] + (correction[1, best_ind]-4) * MAP['res']
    #pose_best_particle[1] = pose_best_particle[1] + (correction[0, best_ind]-4) * MAP['res']
    # Resample
    if 1/np.sum(W2resample*W2resample) <= N_particle/5:
         state[j + 1] = model.resample(W2resample.T, pose_t1, N_particle)
         weight[:, j+1] = 1 / N_particle
    else:
        weight[:, j + 1] = W2resample
        state[j + 1] = pose_t1
    # Select the best particle
    current_scan = model.mapupdate(pose_best_particle, MAP, angles, ranges, confidence)  # update the map
    MAP['map'] = MAP['map'] + current_scan
    MAP['map'][np.where(MAP['map'] > logodds_max)] = logodds_max  # constraint the log-odds within an threshold
    MAP['map'][np.where(MAP['map'] < logodds_min)] = logodds_min
    particle_map[j+1] = MAP['map']
    # Writing data
    update_map[j+1] = current_scan
    plt.clf()
    tract_c[0, j+1] = np.ceil(pose_best_particle[0] * (1 / MAP['res']) + MAP['center'])
    tract_c[1, j+1] = np.ceil(pose_best_particle[1] * (1 / MAP['res']) + MAP['center'])
    tract_xy[:, j+1] = pose_best_particle
    plt.scatter(tract_c[1, 0:j+1], tract_c[0, 0:j+1], c='b', linewidths=0.1)
    plt.imshow(model.log2binary(MAP['map']), cmap='hot')
    plt.savefig(os.getcwd()+'\Results\dataset=%d\%d' %(dataset , j))
    plt.title('Current iteration=%d, Total update time=%f, Expected time=%d minutes' % (j, time.time() - tb, (time.time() - ts)*len(stamps)/60))
    plt.pause(0.000001)

    '''
    if (stamps[j].squeeze() <= rgb_stamps[-1].squeeze()) and (stamps[j].squeeze() >= rgb_stamps[0].squeeze()):
        indx_r = np.argmin(rgb_stamps - stamps[j])
        indx_d = np.argmin(disp_stamps - stamps[j])
        img = Image.open(os.getcwd() + '\\dataRGBD\\Disparity' + str(dataset) + '\\disparity' + str(dataset) + '_' + str(indx_d + 1) + '.png')
        disparity_img = np.array(img.getdata(), np.uint16).reshape(img.size[1], img.size[0])
        RGB = cv2.imread(os.getcwd() + '\\dataRGBD\\RGB' + str(dataset) + '\\rgb' + str(dataset) + '_' + str(indx_d + 1) + '.png')
        tex_map = texture.texturing(tract_xy[:, j+1], RGB, MAP, disparity_img, tex_map)
        plt.imshow(tex_map)
        plt.pause(0.0001)
    te = time.time()
    print('Total update took: %s sec.\n' % (te-ts))
    '''
    toc(ts, " update")
    view_bar(j, len(stamps))



# Save the trajectory for best particle state
np.savez("trajectory%d.npz" % dataset, xy=tract_xy, cell=tract_c)

plt.figure(figsize=(20, 10))
plt.plot(tract_c[1, 1:2000], tract_c[0, 1:2000], c='b', linewidth=3.0)
plt.imshow(model.log2binary(particle_map[2000]), cmap='hot')
plt.title('SLAM at time step = 2000 for dataset%d'%dataset)
plt.savefig('t=2000_data=%d' % dataset)
plt.figure(figsize=(20, 10))
plt.plot(tract_c[1, 1:len(stamps)-1], tract_c[0, 1:len(stamps)-1], c='b', linewidth=3.0)
plt.imshow(model.log2binary(particle_map[len(stamps)-1]), cmap='hot')
plt.title('SLAM at time step = %d for dataset%d'%(len(stamps)-1, dataset))
plt.savefig('t=final_data=%d' % dataset)


