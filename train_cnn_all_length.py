import sys
import numpy as np

from Bio import AlignIO

import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import  Adam

import read_sq2 as read_sq
import cnn_model3

def mahalanobis_dist(test_x, m, train_x, train_y, nclass):
    #calculate mahalanobis distance of test_x1 from classes in training data
    #test_x: test data with which scores are calculated
    #m: model outputs panultimate layer
    #train_x
    #train_y
    #nclass
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

#Read in-group samples
##Read fasta file
alig = AlignIO.read(sys.argv[1], "fasta")
print (alig.get_alignment_length())
print (len(alig))

ealig0 = read_sq.encode_alignment(alig)
print (ealig0.shape)

sqn_nam = [s.id for s in alig]
sqn_nam = np.array(sqn_nam)
print (sqn_nam.shape)
# exit()

##Read species table
samp_nam = []
samp_class = []
with open(sys.argv[2], "r") as f:
    for line in f:
        l = line.rstrip("\n").split(",")
        samp_nam.append(l[0])
        samp_class.append(int(l[2])-1)
        # samp_class.append(int(l[3])-1)

samp_nam = np.array(samp_nam)
samp_class = np.array(samp_class)
samp_class_c = tf.keras.utils.to_categorical(samp_class)

print (samp_nam)
print (samp_class_c.shape)
print (np.all(samp_nam == sqn_nam))

##Read out-of-distribution samples
alig_o = AlignIO.read(sys.argv[3], "fasta")
ealig_o0 = read_sq.encode_alignment(alig_o)
# print (m.predict(ealig_o))
# print (np.argmax(m.predict(ealig_o), axis=1))

o_nam = []
o_class = []
with open(sys.argv[4], "r") as f:
    for line in f:
        l = line.rstrip("\n").split(",")
        o_nam.append(l[0])
        o_class.append(int(l[2])-1)
        # o_class.append(int(l[3])-1)
print (o_class)

if len(sys.argv) >= 6:
    run_code = sys.argv[5]
else:
    run_code = ""

#Model construction
nclass = samp_class_c.shape[1]
# sqlength = ealig.shape[1]

# exit()
#Training with varying sequence lengths
for l in [650, 300, 150]:
    if l == 650:
        ealig = ealig0
        ealig_o = ealig_o0
    else:
        ealig = ealig0[:,350:(350+l),:]
        #ealig = ealig0[:,50:(50+l),:]
        ealig_o = ealig_o0[:,350:(350+l),:]
        #ealig_o = ealig_o0[:,50:(50+l),:]

    sqlength = ealig.shape[1]

    # continue
    for k in range(20):
        print (nclass, sqlength)
        print (ealig.shape, ealig_o.shape)
        # continue
        m = cnn_model3.initialize_dna_cnn_model(sqn_length=sqlength, nclass=nclass, filt2=128, drconv=0.15, drfc=0.25)
        print (m.summary())
        m_pu = Model(m.input, m.layers[-3].output)
        m_nsm = Model(m.input, m.layers[-2].output)
        # print (m_nsm.summary())

        train_size = int(samp_class_c.shape[0]*0.7)
        test_size = samp_class_c.shape[0] - train_size
        #train_size = train_size//2 #Reduce training size ##CHECK THIS###
        print ("train:{0} test:{1}".format(train_size, test_size))
        #exit()
        #train_x, test_x, train_y, test_y = train_test_split(ealig,samp_class_c, test_size=test_size, train_size=train_size)
        train_x, test_x, train_y, test_y = train_test_split(ealig,samp_class_c, test_size=test_size, train_size=train_size, stratify=np.argmax(samp_class_c, 1))

        #Training Model 1
        m.compile(optimizer=Adam(amsgrad=True), loss="categorical_crossentropy", metrics=["accuracy"])
        #m.fit(train_x, train_y, epochs=400, verbose=1, validation_data=(test_x, test_y))
        m.fit(train_x, train_y, epochs=500, verbose=1, validation_data=(test_x, test_y))
        #m.fit(train_x, train_y, epochs=700, verbose=1, validation_data=(test_x, test_y))
        #m.fit(train_x, train_y, epochs=800, verbose=1, validation_data=(test_x, test_y))

        #Output results
        # print(np.sort(np.argmax(m.predict(test_x), axis=1)))
        # print(np.sort(np.argmax(test_y, axis=1)))
        print ("training acc1:")
        print(sum(np.argmax(m.predict(train_x), axis=1) == np.argmax(train_y, axis=1))/len(train_y))
        print ("test acc1:")
        print(sum(np.argmax(m.predict(test_x), axis=1) == np.argmax(test_y, axis=1))/len(test_y))

        f_nsm = m_nsm.predict(test_x)
        enrg = -np.log(np.sum(np.exp(f_nsm), axis=1))
        print (np.mean(enrg))

        mah_d = mahalanobis_dist(test_x, m_pu, train_x, train_y, nclass)
        u1 = np.min(mah_d, axis=1) 

        test_class = np.argmax(test_y, axis=1)
        pred_class1 = np.argmax(m.predict(test_x), axis=1)
        pred_prob = np.max(m.predict(test_x), axis=1)
        pred_class2 = np.argmin(mah_d, axis=1)#np.argmax(m_d.predict(test_x), axis=1)
        with open("pred.prob.metrics.{0}.varlen.txt".format(run_code), "a") as f:
            if k == 0 and l == 650:
                f.write("length\tk\ttrue_class\tpred_class1\tprob\tenergy\tpred_class2\tuncertainty\n")
            for l1, l2, p, e, l3, u in zip(test_class, pred_class1, pred_prob, enrg, pred_class2, u1):
                #print ("{0}\t{1}\t{2}".format(l1, l2, e))
                f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n".format(l, k, l1, l2, p, e, l3, u))

        ###test with ood samples
        print (np.argmax(m.predict(ealig_o), axis=1))

        f_nsm_o = m_nsm.predict(ealig_o)
        enrg = -np.log(np.sum(np.exp(f_nsm_o), axis=1))
        print (np.mean(enrg))

        mah_d_o = mahalanobis_dist(ealig_o, m_pu, train_x, train_y, nclass)
        u2 = np.min(mah_d_o, axis=1)

        test_class = -np.ones(len(ealig_o))
        pred_class1 = np.argmax(m.predict(ealig_o), axis=1)
        pred_prob = np.max(m.predict(ealig_o), axis=1)
        pred_class2 = np.argmin(mah_d_o, axis=1)#np.argmax(m_d.predict(ealig_o), axis=1)
        with open("pred.prob.metrics.ood.{0}.varlen.txt".format(run_code), "a") as f:
            if k == 0 and l == 650:
                f.write("length\tk\ttrue_class\tpred_class1\tprob\tenergy\tpred_class2\tuncertainty\n")
            for l1, l2, p, e, l3, u in zip(test_class, pred_class1, pred_prob, enrg, pred_class2, u2):
                #print ("{0}\t{1}\t{2}".format(l1, l2, e))
                f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n".format(l, k, l1, l2, p, e, l3, u))
