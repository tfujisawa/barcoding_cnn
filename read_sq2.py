import sys
import numpy as np

from Bio import SeqIO
from Bio import AlignIO

base_index = {'A':0, 'T':1, 'G':2, 'C':3}

def onehot_dna00(sqn):
    v = np.zeros((len(sqn), 4))
    for i, b in enumerate(sqn):
        if b != 'N' and b != "-" and b!= "S" and  b != "M" and  b != "R" and b != "Y" and b != "K" and b != "W" and b != "D" and b != "B" and b != "V":
            v[i,base_index[b]] = 1

    return (v)

def encode_alignment(alig):
    encoded_dna = np.zeros((len(alig), alig.get_alignment_length(), 4))
    for i in range(len(alig)):
        encoded_dna[i,:,:] = onehot_dna00(alig[i,:].upper())
    return (encoded_dna)

nbase_index = {'A':[1,2,3], 'T':[0,2,3], 'G':[0,1,3], 'C':[0,1,2]}

def onehot_dna00_withError(sqn, e=0.05):
    v = np.zeros((len(sqn), 4))
    for i, b in enumerate(sqn):
        if b != 'N' and b != "-" and b!= "S" and  b != "M" and  b != "R" and b != "Y" and b != "K" and b != "W" and b != "D" and b != "B" and b != "V":
            if np.random.sample() < e:
                v[i,np.random.choice(nbase_index[b], 1)] = 1
            else:
                v[i,base_index[b]] = 1
    return (v)

def encode_alignment_withError(alig, n=1, e=0.05):
    for k in range(n):
        encoded_dna = np.zeros((len(alig), alig.get_alignment_length(), 4))
        for i in range(len(alig)):
            encoded_dna[i,:,:] = onehot_dna00_withError(alig[i,:].upper(), e=e)
        yield (encoded_dna)

if __name__ == "__main__":
    alig = AlignIO.read(sys.argv[1], "fasta")
    print (alig.get_alignment_length())
    print (len(alig))

    ealig = encode_alignment(alig)
    print (ealig.shape)
    print (ealig[0][40:50,:])

    ealige = encode_alignment_withError(alig, e=0.2, n=3)
    for i, a in enumerate(ealige):
        print (i)
        print (a[0][40:50,:])

