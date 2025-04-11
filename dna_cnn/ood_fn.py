import numpy as np

from tensorflow.keras.models import Model

def energy(test_x, m):
    #calculate energy scores of test_x with model m
    #test_x: test data with which scores are calculated
    #m: model outputs of the final FC layer WITHOUT Softmax

    f_nsm = m.predict(test_x)
    enrg = -np.log(np.sum(np.exp(f_nsm), axis=1))
    return (enrg)

def mahalanobis_dist(test_x, m, train_x, train_y, nclass):
    #calculate mahalanobis distance of test_x1 from class centers in training data with model m
    #test_x: test data with which scores are calculated
    #m: model outputs panultimate layer
    #train_x: training features
    #train_y: training labels
    #nclass: the number of classes
    f_pu_t = m.predict(train_x)
    part_y = np.argmax(train_y, 1)
    sigm = np.array([np.cov(f_pu_t[part_y==i].T) for i in range(nclass)])
    sigm = np.mean(sigm, axis=0) #tied covariance matrix
    # print(sigm.shape)
    # print(sigm)
    # print(np.linalg.det(sigm))
    # sigm_inv = np.linalg.inv(sigm)
    sigm_inv = np.linalg.pinv(sigm)

    mu_c = [np.mean(f_pu_t[part_y==i,], axis=0) for i in range(nclass)]
    mu_c = np.array(mu_c)

    f_pu = m.predict(test_x)

    res = []
    for i in range(len(f_pu)):
        d = [(f_pu[i,:] - mu_c[k,:])@sigm_inv@(f_pu[i,:]-mu_c[k,:]).T for k in range(nclass)]
        #mah_pred.append(np.argmin(d))
        #d = np.min(d)
        res.append(d)
    return (np.array(res))


