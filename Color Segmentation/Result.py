import os, cv2, math
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt


class BarrelDetector():
    def __init__(self):
        import pickle
        f = open('alpha_rgb.data', 'rb')
        self.alpha = pickle.load(f)
        f.close()
        f = open('sigma_rgb.data', 'rb')
        self.sigma = pickle.load(f)
        f.close()
        f = open('mu_rgb.data', 'rb')
        self.mu = pickle.load(f)
        f.close()
        f = open('theta_rgb.data', 'rb')
        self.theta = pickle.load(f)
        f.close()

    def segment_image(self, img):
        from scipy.stats import multivariate_normal
        N_cluster = 4
        N_class = 3
        test_vec = np.reshape(img, (-1, 3))
        prob = [np.log(self.theta[k]) + np.log(
            sum([self.alpha[k][l] * multivariate_normal.pdf(test_vec, self.mu[k][l], self.sigma[k][l]) \
                 for l in range(N_cluster)])) for k in range(N_class)]
        y = np.reshape([[prob[0][k], prob[1][k], prob[2][k]].index(max([prob[0][k], prob[1][k], prob[2][k]])) for k in
                        range(800 * 1200)], (800, 1200))
        mask_img = np.where(y > 1, 1, 0)
        return mask_img

    def get_bounding_box(self, img):
        boxes = []
        kernel = np.ones((3, 3), np.uint8)
        img_mask = self.segment_image(img)
        mask = cv2.erode(img_mask.astype(np.float32), kernel, iterations=4)
        mask_img = cv2.dilate(mask.astype(np.float32), kernel, iterations=6)
        label_img = label(mask_img)
        regions = regionprops(label_img)
        for props in regions:
            if props.area > 750:
                boxes.append([props.bbox[1], props.bbox[0], props.bbox[3], props.bbox[2]])
        return boxes


image = []
CWD = os.getcwd()
N_trainset = 46
for i in range(N_trainset):
    image.append(cv2.imread(CWD + '/trainset/' + np.str(i+1) + '.png'))
barrel = BarrelDetector()
for i in range():
    plt.figure()
    plt.imshow(cv2.cvtColor(image[i], cv2.COLOR_RGB2BGR ))
    boxes = barrel.get_bounding_box(image[i])
    for box in boxes:
        [minc, minr, maxc, maxr]=box
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        plt.plot(bx, by, '-r', linewidth=2.5)
    plt.figure()
    plt.imshow(barrel.segment_image(image[i]))N_trainset