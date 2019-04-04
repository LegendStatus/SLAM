import numpy as np
from data import map_utils as mp


# Weight resample
def resample(weight, pos, N):
    '''
    :param weight: 1-D array containing weight of each particle
    :param pos: 2-D array containing state of each particle
    :param n: Number of total particle
    :return:
    '''
    pos_resampled = np.zeros((3, N))
    weight_resampled = np.zeros((N, 1))
    j = 0
    c = weight[0]
    u = 1 / N
    for k in range(N):
        beta = u + k/N
        while beta - c > 0.000001:
            j += 1
            c += weight[j]
        pos_resampled[:, k] = pos[:, j]
        weight_resampled[k] = 1/N
    return pos_resampled


# Map update
def mapupdate(state, MAP, angles, ranges, confidence):
    '''
    :param state: state of best particle in world frame in cell position
    :param MAP: current log-odds map
    :param lidar_x: filtered x-position of lidar scan in lidar frame
    :param lidar_y: filtered y-position of lidar scan in lidar frame
    :return: updated log-odds map
    '''
    map = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)
    theta = state[2]; x = state[0]; y = state[1]  # Reading state
    # Selecting valid range
    indValid = np.logical_and((ranges < 30), (ranges > 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    offset_y = (301.83 - 330.2 / 2) / 1000
    # To World frame
    lidar_xW = (ranges * np.cos(angles + theta) - np.sin(theta) * offset_y + x) / MAP['res']
    lidar_yW = (ranges * np.sin(angles + theta) + np.cos(theta) * offset_y + y) / MAP['res']
    # Lidar frame to body frame which is also world frame in initialization
    cent_xW = (0 * np.cos(theta) - np.sin(theta) * offset_y + x) / MAP['res']
    cent_yW = (0 * np.sin(theta) + np.cos(theta) * offset_y + y) / MAP['res']

    # Obtaining occupied and free grid
    free = [np.int32(mp.bresenham2D(cent_xW, cent_yW, lidar_xW[i], lidar_yW[i]) + MAP['center']) for i in range(len(ranges))]
    ocy = np.int32(np.vstack((np.round(lidar_xW), np.round(lidar_yW))) + MAP['center'])
    # Add log-odds to the grid in MAP
    for i in range(len(ranges)):
        map[tuple(free[i])] = -np.log(confidence)
        map[tuple(ocy[:, i])] = np.log(confidence)
    return map


# Weight update
def bestparticle(pos):
    '''
    :param pos: set of pose of all particles
    :return: the column index of the particles that appears most
    '''
    dict = pos.T.tolist()
    cot = [0] * len(dict)
    for i in range(len(dict)):
        cot[i] = dict.count(dict[i])
    return cot.index(max(cot))


# Map initialization
def createmap(angles, ranges, confidence):
    # Mapping Initialization test
    MAP = {}
    MAP['res'] = 0.1  # meters
    MAP['xmin'] = -30  # meters
    MAP['ymin'] = -30
    MAP['ymax'] = 30
    MAP['xmax'] = 30
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)  # DATA TYPE: char or int8
    MAP['center'] = (MAP['xmax']-MAP['xmin'])/(2*MAP['res'])  # Center cell position
    # Occupancy grid information
    indValid = np.logical_and((ranges < 30), (ranges > 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    xs0 = ranges * np.cos(angles) / MAP['res']
    ys0 = ranges * np.sin(angles) / MAP['res']
    # Lidar frame to body frame which is also world frame in initialization
    offset_y = np.ceil((301.83 - 330.2 / 2) / 1000 / MAP['res'])
    lidar_y = ys0 + offset_y
    lidar_x = xs0
    # Obtain the free and occupied grid
    free = [np.int32(mp.bresenham2D(0, 0, lidar_x[i], lidar_y[i]) + MAP['center']) for i in range(len(lidar_x))]
    ocy = np.int32(np.vstack((np.round(lidar_x), np.round(lidar_y))) + MAP['center'])
    # Add log-odds to the grid in MAP
    for i in range(len(lidar_x)):
        MAP['map'][tuple(free[i])] = -np.log(confidence)
        MAP['map'][tuple(ocy[:, i])] = np.log(confidence)
    return MAP


# Time stamps coordination
def cor(objdata, objstamps, target):
    '''
    target = 1-D array time stamps to be coordinated with
    objdata = 1-D array data that need to be coordinated with target
    objstamps = 1-D array time stamps of object
    '''
    l = len(target)
    target = target - target[0] # normalized time stamps starting from 0
    objstamps = objstamps - objstamps[0]
    cor_obj = np.zeros((l,1))
    cor_obj[0] = objdata[0] # initialization
    for i in range(l-1):
        lower_limit = np.where(objstamps < target[i+1])[0][-1]
        if lower_limit == len(objstamps)-1:
            upper_limit = lower_limit
            slope = 0
        else:
            upper_limit = lower_limit + 1
            slope = (objdata[upper_limit] - objdata[lower_limit])/(objstamps[upper_limit] - objstamps[lower_limit])
        cor_obj[i+1] = objdata[lower_limit] + (target[i+1] - objstamps[lower_limit]) * slope
    return cor_obj

# Motion Model
def motion(v_t, w_t, x_t, dt):
    '''
    Inputs: 
        v_t = plane linear velocity of center os gravity at time t
        w_t = yaw rate of center os gravity at time t
        x_t = state at time t
        dt = time stamps
    Outputs:
        x_t1 = state at time t+1
    '''
    theta = x_t[2]
    # L = (476.25-311.15)/2+311.15
    x_t1 = x_t + (dt*np.float64([v_t*np.sinc(w_t*dt/2/np.pi)*np.cos(theta+w_t*dt/2),\
                     v_t*np.sinc(w_t*dt/2/np.pi)*np.sin(theta+w_t*dt/2), w_t])).T
    return x_t1



def softmax(x):
    '''
    :param x: 1-D numpy array
    :return: corresponding softmax array
    '''
    x_norm = x - np.max(x)  # normalization
    s = np.exp(x_norm) / np.sum(np.exp(x_norm))
    return s


def log2binary(MAP):
    m = 1 - 1/(1 + np.exp(MAP))
    map = np.zeros(MAP.shape)  # represent the rest of occupancy and free grid with 0
    map[np.where(m > 0.95)] = 2
    map[np.where(m < 0.05)] = 1
    return map


# os = np.array([[1,2,3,4], [3,4,5,6], [4,5,6,7]])
# wei = np.array([0.7, 0.2, 0.05, 0.05])
# pr, wr = resample(wei.T, pos, 4)

