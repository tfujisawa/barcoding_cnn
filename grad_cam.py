import sys
import numpy as np

from Bio import AlignIO

import tensorflow as tf

from tensorflow.keras.models import Model

import read_sq2 as read_sq
import cnn_model3

def grad_cam(model, img_input, last_conv_layer, pred_layers):
    lclayer = model.get_layer(last_conv_layer)
    lcmodel = Model(model.inputs, lclayer.output)

    pred_input = tf.keras.Input(shape=lclayer.output.shape[1:])
    x = pred_input
    for n in pred_layers:
        x = model.get_layer(n)(x)
    pred_model = Model(pred_input, x)

    with tf.GradientTape() as tape:
        lc_out = lcmodel(img_input)
        tape.watch(lc_out)
        # print (lclayer.output.shape)
        # pred = pred_model(tf.reshape(lc_out, (1,85,32))) #CHange this to fit input dimension
        pred = pred_model(tf.reshape(lc_out, (1, lclayer.output.shape[1], lclayer.output.shape[2]))) 
        top_pred = tf.argmax(pred[0])
        top_class = pred[:,top_pred]

    grad = tape.gradient(top_class, lc_out)
    pooled_grad = tf.reduce_mean(grad, axis=(0,1))

    out = lc_out.numpy()[0]
    pooled_grad = pooled_grad.numpy()
    for i in range(pooled_grad.shape[-1]):
        out[:,i] *= pooled_grad[i]

    mout = np.mean(out, axis=1)
    mout = np.maximum(mout, 0)/np.max(mout)

    return (mout)

if __name__ == "__main__":
    m = tf.keras.models.load_model(sys.argv[1])##a model
    print (m.summary())
    lcl = m.layers[-8].name#"max_pooling1d_2" #lcl : last convolution layer in string
    prl = [n.name for n in m.layers[-7:]] #prl: prediction layers in a list of strings

    alig = AlignIO.read(sys.argv[2], "fasta")
    print (alig.get_alignment_length())
    print (len(alig))

    l = m.input.shape[1]
    print (l)
    ealig = read_sq.encode_alignment(alig)
    if l < 650:
        ealig = ealig[:,350:(350+l)] #partial seq
    print (ealig.shape)
    
    sqn_nam = [s.id for s in alig]
    sqn_nam = np.array(sqn_nam)
    print (sqn_nam.shape)

    #grad_cam(m, test_x[0:1,:], lcl, prl)
    #test_class = np.argmax(test_y, axis=1)
    pred_class1 = np.argmax(m.predict(ealig), axis=1)
    print (pred_class1)
    #exit()
    outfile = "grad.sqn.{0}.txt".format(sys.argv[1].rstrip("\/"))
    print ("writing ", outfile, " ...")
    with open(outfile, "w") as f:
        for i in range(len(ealig)):
            g = grad_cam(m, ealig[i:(i+1),:], lcl, prl)
            #print ("{0},{1},".format(sqn_nam[i], pred_class1[i])+",".join([str(s) for s in g]))
            f.write("{0},{1},".format(sqn_nam[i], pred_class1[i])+",".join([str(s) for s in g]) + "\n")
