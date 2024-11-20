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

if __name__ == "__main__":
    alig = AlignIO.read(sys.argv[1], "fasta")
    print (alig.get_alignment_length())
    print (len(alig))

    ealig = encode_alignment(alig)
    print (ealig.shape)
