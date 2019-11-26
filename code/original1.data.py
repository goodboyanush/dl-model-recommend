import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path="", dicty=None):
        if(len(path) > 0):
            self.dictionary = Dictionary()
            self.train = self.tokenize_path(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize_path(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize_path(os.path.join(path, 'test.txt'))
        self.dictionary = dicty

    def tokenize_path(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def tokenize_sentence(self, sentence):
        # Tokenize sentence content
        words = sentence.split() + ['<eos>']
        tokens = len(words)
        
        #ids = torch.LongTensor(len(self.dictionary.idx2word))
        ids = torch.LongTensor(tokens)
        token = 0
        for word in words:
            ids[token] = self.dictionary.word2idx[word]
            token += 1

        return ids