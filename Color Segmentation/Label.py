import numpy as np
import os, cv2, pickle
import matplotlib.pyplot as plt
from roipoly import RoiPoly

CWD = os.getcwd()
N_trainset = 46
image = []
# Represent the first blue barrel
roi_barrel_1 = [None]*N_trainset
roi_barrel_2 = [None]*N_trainset
# Represent the first blue but not barrel staff
roi_nonbarrel_1 = [None]*N_trainset
roi_nonbarrel_2 = [None]*N_trainset

# Read the image data
for i in range(N_trainset):
    image.append(cv2.imread(CWD + '/trainset/' + np.str(i+1) + '.png'))

for i in range(N_trainset):
    plt.imshow(image[i])
    roi_barrel_1[i] = RoiPoly(color='r')
    plt.imshow(image[i])
    roi_barrel_2[i] = RoiPoly(color='r')
    plt.imshow(image[i])
    roi_nonbarrel_1[i] = RoiPoly(color='r')
    plt.imshow(image[i])
    roi_nonbarrel_2[i] = RoiPoly(color='r')


def label_binary(roi_1, roi_2):
    (l, w) = cv2.split(image[0])[0].shape
    label = [np.zeros((l, w))] * N_trainset
    mask_barrel_1 = [None] * N_trainset
    mask_barrel_2 = [None] * N_trainset
    for j in range(N_trainset):
        mask_barrel_1[j] = roi_1[j].get_mask(cv2.split(image[j])[0])
        mask_barrel_2[j] = roi_2[j].get_mask(cv2.split(image[j])[0])
        mask = (mask_barrel_1[j]+0) + (mask_barrel_2[j]+0)
        label[j] = np.where(mask == 0, -1, 1)
    f = open('label_binary.data1', 'wb')
    pickle.dump(label, f)
    f.close()


def label_K_class(roi_barrel_1, roi_barrel_2, roi_nonbarrel_1, roi_nonbarrel_2):
    (l, w) = cv2.split(image[0])[0].shape
    label = [np.zeros((l, w))] * N_trainset
    mask_barrel_1 = [None] * N_trainset
    mask_barrel_2 = [None] * N_trainset
    mask_nonbarrel_1 = [None] * N_trainset
    mask_nonbarrel_2 = [None] * N_trainset
    for j in range(N_trainset):
        mask_barrel_1[j] = roi_barrel_1[j].get_mask(cv2.split(image[j])[0])
        mask_barrel_2[j] = roi_barrel_2[j].get_mask(cv2.split(image[j])[0])
        mask_nonbarrel_1[j] = roi_nonbarrel_1[j].get_mask(cv2.split(image[j])[0])
        mask_nonbarrel_2[j] = roi_nonbarrel_2[j].get_mask(cv2.split(image[j])[0])
        mask = (mask_barrel_1[j]+0) + (mask_barrel_2[j]+0)
        label[j] = np.where(mask == 0, -1, 1)
        label[j] = label[j] + ((mask_nonbarrel_1[j]+0) + (mask_nonbarrel_2[j]+0))
        label[j][np.where(label[j] > 1)] = 1
        label[j] = label[j] + 2
    f = open('label_K_class.data1', 'wb')
    pickle.dump(label, f)
    f.close()


label_binary(roi_barrel_1, roi_barrel_2)
label_K_class(roi_barrel_1, roi_barrel_2, roi_nonbarrel_1, roi_nonbarrel_2)
