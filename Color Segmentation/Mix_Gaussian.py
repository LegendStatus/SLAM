import pickle, cv2, random, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

# Read Image files
image = []
CWD = os.getcwd()
N_trainset = 46; N_class = 3; dim = 3
for i in range(N_trainset):
    image.append(cv2.imread(CWD + '/trainset/' + np.str(i+1) + '.png')   )


# Read the hand-label data
f = open('label_K_class.data', 'rb')
mask_k_class = pickle.load(f)
f.close()

# Pick randomly 40 images as training set and the other 6 images as validation set
Ind = list(np.arange(N_trainset))
random.shuffle(Ind)
Ind_train = Ind[0:40]; Ind_validation = Ind[40:N_trainset]

# Reshape label mask and image to vectors
img_train = np.reshape(image[Ind_train[0]], (-1, 3))
label_train = np.reshape(mask_k_class[Ind_train[0]], (-1, 1))
for i in range(1, len(Ind_train)):
    label_train = np.vstack((label_train, np.reshape(mask_k_class[Ind_train[i]], (-1, 1))))
    img_train = np.vstack((img_train, np.reshape(image[Ind_train[i]], (-1, 3))))

# Training
theta = []; alpha = []; mu = []; sigma = []; N_cluster = 4
for k in range(N_class):
    theta.append(len(label_train[np.where(label_train == k+1)])/len(label_train))
# Initialization for parameters of mixture gaussian model using k-means strategy
for k in range(N_class):
    temp_X = img_train[np.transpose(np.where(label_train == k+1)[0])]
    kmeans = KMeans(N_cluster, max_iter=15, random_state=0).fit(temp_X)
    mu.append(kmeans.cluster_centers_)
    temp_alpha = []; temp_sigma = []
    for j in range(N_cluster):
        temp_alpha.append(len(temp_X[np.where(kmeans.labels_ == j)])/len(temp_X[:, 0]))
        temp_sigma.append(np.cov(np.transpose(temp_X[np.transpose(np.where(kmeans.labels_ == j))[:, 0]])))
    alpha.append(temp_alpha); sigma.append(temp_sigma)


# Expectation Maximization
for iteration in range(5):
    # E step
    r=[]
    for k in range(N_class):
        temp_r=[]
        for j in range(N_cluster):
            temp_r.append( alpha[k][j] * multivariate_normal.pdf(img_train, mu[k][j], sigma[k][j]) \
                        /sum([alpha[k][l] * multivariate_normal.pdf(img_train, mu[k][l], sigma[k][l]) \
                                for l in range(N_cluster)]) )
        r.append(temp_r)
    # M step
    for k in range(N_class):
        Ind_X = np.where(label_train == k + 1)[0]
        for j in range(N_cluster):
            alpha[k][j] = np.sum(r[k][j][Ind_X])/len(Ind_X)
            mu[k][j] = np.dot(np.transpose(img_train[Ind_X]), r[k][j][Ind_X])/np.sum(r[k][j][Ind_X])
            temp = (img_train[Ind_X]-mu[k][j]).T * np.asarray([r[k][j][Ind_X]])
            sigma[k][j] = np.dot(temp, (img_train[Ind_X]-mu[k][j]))/np.sum(r[k][j][Ind_X])


f = open('theta_rgb.data', 'wb')
pickle.dump(theta, f)
f.close()
f = open('alpha_rgb.data', 'wb')
pickle.dump(alpha, f)
f.close()
f = open('mu_rgb.data', 'wb')
pickle.dump(mu, f)
f.close()
f = open('sigma_rgb.data', 'wb')
pickle.dump(sigma, f)
f.close()


# Testing
def color_segement(img):
    test = img
    test_vec = np.reshape(test,  (-1, 3))
    prob = [np.log(theta[k]) + np.log(sum([alpha[k][l] * multivariate_normal.pdf(test_vec, mu[k][l], sigma[k][l]) \
                                for l in range(N_cluster)])) for k in range(N_class)]
    y = np.reshape( [[prob[0][k], prob[1][k], prob[2][k]].index(max([prob[0][k], prob[1][k], prob[2][k]])) \
                     for k in range(800*1200)], (800, 1200))
    mask_img = np.where(y > 1, 1, 0)
    return mask_img


for x in Ind_validation:
    plt.figure()
    plt.imshow(color_segement(image[x]))






