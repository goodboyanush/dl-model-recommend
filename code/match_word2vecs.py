import numpy as np
import torch
import pickle
word2vec = {}

def getWordvectors(filename,vocabfile='../../models/vocab_w2v.txt'):
    word2vec = {}
    f = open(filename,'r')
    fv = open(vocabfile,'r')
    vocab = [line.strip() for line in fv]
    w2vsize = 300
    for line in f.readlines():
        word, vector = line.split(' ',1)
        word = word.strip()
        vector = np.fromstring(vector, dtype=float, sep=' ')
        w2vsize = vector.flatten().shape[0]
        word2vec[word] = vector
    for word in vocab:
        if word not in word2vec.keys():
            word2vec[word] = np.random.rand(w2vsize)
    return word2vec

def load_embeddings(embedDim, vocabSize, word2vec, dictionary):
    emb = torch.FloatTensor(vocabSize, embedDim).normal_(-0.05,0.05)

    for word in sorted(dictionary.word2idx.keys()):
        assert(word in word2vec.keys() or ['<pad>', '<unk>', '<bos>', '<eos>'])
        if word not in ['<pad>', '<unk>', '<bos>', '<eos>']:
            if word in word2vec.keys():
                emb[dictionary.word2idx[word]] = torch.from_numpy(word2vec[word])
        else:
        	emb[dictionary.word2idx[word]].zero_()
    return emb
