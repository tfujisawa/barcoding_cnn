import sys
import numpy as np

from Bio import AlignIO

import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import  Adam

import dna_cnn.read_sq2 as read_sq
import dna_cnn.cnn_model3 as cnn_model3

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

    print (nclass, sqlength)
    print (ealig.shape, ealig_o.shape)
    # continue
    
    m = cnn_model3.initialize_dna_cnn_model(sqn_length=sqlength, nclass=nclass, filt2=128, drconv=0.15, drfc=0.25)
    print (m.summary())
    #m_pu = Model(m.input, m.layers[-3].output)
    m_nsm = Model(m.input, m.layers[-2].output)

    train_size = int(samp_class_c.shape[0]*0.7)
    test_size = samp_class_c.shape[0] - train_size
    
    print ("train:{0} test:{1}".format(train_size, test_size))
    
    #train_x, test_x, train_y, test_y = train_test_split(ealig,samp_class_c, test_size=test_size, train_size=train_size)
    train_x, test_x, train_y, test_y = train_test_split(ealig,samp_class_c, test_size=test_size, train_size=train_size, stratify=np.argmax(samp_class_c, 1))

    #Training Model
    m.compile(optimizer=Adam(amsgrad=True), loss="categorical_crossentropy", metrics=["accuracy"])
    m.fit(train_x, train_y, epochs=400, verbose=1, validation_data=(test_x, test_y))

    #Output results
    print ("training acc1:")
    print(sum(np.argmax(m.predict(train_x), axis=1) == np.argmax(train_y, axis=1))/len(train_y))
    print ("test acc1:")
    print(sum(np.argmax(m.predict(test_x), axis=1) == np.argmax(test_y, axis=1))/len(test_y))

    f_nsm = m_nsm.predict(test_x)
    enrg = -np.log(np.sum(np.exp(f_nsm), axis=1))
    print ("average ID energy:")
    print (np.mean(enrg))

    f_nsm_o = m_nsm.predict(ealig_o)
    enrg_o = -np.log(np.sum(np.exp(f_nsm_o), axis=1))
    print ("average OOD energy:")
    print (np.mean(enrg_o))

    model_nam = "model.{0}.{1}.keras".format(l, run_code)
    print ("saving " + model_nam)
    tf.keras.models.save_model(m, model_nam)
