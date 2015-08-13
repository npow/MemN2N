from __future__ import division
import argparse
import glob
import lasagne
import numpy as np
import sys
from sklearn import metrics
from sklearn.feature_extraction.text import *
from sklearn.preprocessing import *
import theano
import theano.tensor as T

class Model:
    def __init__(self, train_file, test_file, output_size=150):
        self.train_lines, self.test_lines = self.get_lines(train_file), self.get_lines(test_file)
        lines = np.concatenate([self.train_lines, self.test_lines], axis=0)

        self.vectorizer = CountVectorizer(lowercase=False)
        self.vectorizer.fit([x['text'] + ' ' + x['answer'] if 'answer' in x else x['text'] for x in lines])

        X = self.vectorizer.transform([x['text'] for x in lines]).toarray().astype(np.float32)
        self.C_train, self.Q_train, self.A_train, train_max_seqlen = self.get_dataset(self.train_lines, X[:len(self.train_lines)])
        self.C_test, self.Q_test, self.A_test, test_max_seqlen = self.get_dataset(self.test_lines, X[len(self.train_lines):])

        print self.Q_train.shape, self.A_train.shape
        print self.Q_test.shape, self.A_test.shape
        print len(lines), train_max_seqlen, test_max_seqlen
        max_Q_seqlen = max(train_max_seqlen, test_max_seqlen)

        return

        vocab_size = len(self.vectorizer.vocabulary_)

        Q_W = np.arange(vocab_size * embedding_size).reshape((vocab_size, embedding_size)).astype(theano.config.floatX)
        l_Q_in = lasagne.layers.InputLayer(shape=(batch_size, max_Q_len * vocab_size))
        l_Q_embedding = lasagne.layers.EmbeddingLayer(l_Q_in, input_size=vocab_size, output_size=embedding_size, W=Q_W)

        A_W = np.arange(vocab_size * embedding_size).reshape((vocab_size, embedding_size)).astype(theano.config.floatX)
        l_A_in = lasagne.layers.InputLayer(shape=(batch_size, 1 * vocab_size))
        l_A_embedding = lasagne.layers.EmbeddingLayer(l_A_in, input_size=vocab_size, output_size=embedding_size, W=A_W)

        l_prob = InnerProductLayer(l_Q_embedding, l_A_embedding, nonlinearity=lasagne.nonlinearities.softmax)
        l_output = lasagne.layers.DenseLayer(l_Q_embedding, output_size, b=None, nonlinearity=None)
        l_weighted_output = lasagne.layers.ElemWiseMergeLayer(l_prob, l_output, merge_function=T.mul)

        l_sum = lasagne.layers.ElemWiseSumLayer(l_weighted_output, l_A_embedding)
        l_pred = lasagne.layers.DenseLayer(l_sum, vocab_size, nonlinearity=lasagne.nonlinearities.softmax)

    def get_dataset(self, lines, vectorized_lines):
        C, Q, A = [], [], []
        max_seqlen = 0
        for i,line in enumerate(lines):
            if line['type'] == 'q':
                id = line['id']-1
                indices = [idx+1 for idx in range(i-id, i) if lines[idx]['type'] == 's']
                max_seqlen = max(len(indices), max_seqlen)
                C.append(indices)
                Q.append(i+1)
                A.append(line['answer'])
        return C, np.array(Q, dtype=np.int32), self.vectorizer.transform(A), max_seqlen

    def get_lines(self, fname):
        lines = [{'type':'s', 'text':''}]
        for i,line in enumerate(open(fname)):
            id = int(line[0:line.find(' ')])
            line = line.strip()
            line = line[line.find(' ')+1:]        
            if line.find('?') == -1:
                lines.append({'type':'s', 'text': line})
            else:
                idx = line.find('?')
                tmp = line[idx+1:].split('\t')
                lines.append({'id':id, 'type':'q', 'text': line[:idx], 'answer': tmp[1].strip(), 'refs': [int(x) for x in tmp[2:][0].split(' ')]})
            if False and i > 1000:
                break
        return np.array(lines)

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def main():
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.add_argument('--task', type=int, default=1, help='Task#')
    parser.add_argument('--train_file', type=str, default='', help='Train file')
    parser.add_argument('--test_file', type=str, default='', help='Test file')
    args = parser.parse_args()
    print "args: ", args

    train_file = glob.glob('data/en-10k/qa%d_*train.txt' % args.task)[0]
    test_file = glob.glob('data/en-10k/qa%d_*test.txt' % args.task)[0]
    if args.train_file != '' and args.test_file != '':
        train_file, test_file = args.train_file, args.test_file

    model = Model(train_file, test_file)

if __name__ == '__main__':
    main()
