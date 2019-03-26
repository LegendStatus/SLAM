import pickle, cv2, random, os
import numpy as np
import matplotlib.pyplot as plt

# Read Image files
image_rgb = []; image_luv = []
CWD = os.getcwd()
N_trainset = 46 ; N_class = 3; dim = 6
for i in range(N_trainset):
    rgb = cv2.imread(CWD + '/trainset/' + np.str(i+1) + '.png')
    luv = cv2.cvtColor(rgb, cv2.COLOR_BGR2Luv)
    image_rgb.append(rgb); image_luv.append(luv)


# Read the hand-label data
f = open('label_K_class.data', 'rb')
mask_k_class = pickle.load(f)
f.close()

# Pick randomly 40 images as training set and the other 6 images as validation set
Ind = list(np.arange(46))
random.shuffle(Ind)
Ind_train = Ind[0:40]; Ind_validation = Ind[40:46]

# Reshape label mask and image to vectors
label_train = np.reshape(mask_k_class[Ind_train[0]], (-1, 1))
img_rgb_train = np.reshape(image_rgb[Ind_train[0]], (-1, 3))
img_luv_train = np.reshape(image_luv[Ind_train[0]], (-1, 3))
img_train = np.hstack((img_rgb_train, img_luv_train))
for i in range(1, len(Ind_train)):
    label_train = np.vstack((label_train, np.reshape(mask_k_class[Ind_train[i]], (-1, 1))))
    img_train = np.vstack((img_train, np.hstack((np.reshape(image_rgb[Ind_train[i]], (-1, 3)),\
                                                 np.reshape(image_luv[Ind_train[i]], (-1, 3))))))

# Training
def softmax(z):
    sz = sum([np.exp(z[i]) / np.sum(np.exp(z[i]), axis=0) for i in range(N_class)])
    return sz

w = np.zeros((N_class, dim))
learn_rate=10
for iteration in range(20):
    np.eye(N_class)[:,label_train[i]-1]
    w = w + learn_rate*(sum([np.dot((np.eye(N_class)[:,label_train[i]-1] - softmax(w)), img_train[i]) for i in range(len(Ind_train))]))

def color_segement(img):
    test_luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    test_vec = np.hstack((np.reshape(img, (-1, 3)), np.reshape(test_luv, (-1, 3))))
    prob = np.dot(test_vec, w.T)
    y = np.argmax(prob).reshape((800,1200))
    mask_img = np.where(y > 1, 1, 0)
    return mask_img


plt.imshow(color_segement(image_rgb[5]))