from __future__ import division
import argparse
import glob
import lasagne
import numpy as np
import sys
import theano
import theano.tensor as T
import time
from lasagne.layers import MergeLayer
from sklearn import metrics
from sklearn.feature_extraction.text import *
from sklearn.preprocessing import *
from theano.printing import Print as pp

import warnings
warnings.filterwarnings('ignore', '.*topo.*')

class InnerProductLayer(MergeLayer):
    def __init__(self, incomings, nonlinearity=None, **kwargs):
        super(InnerProductLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = nonlinearity
        if len(incomings) != 2:
            raise NotImplementedError

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:2]

    def get_output_for(self, inputs, **kwargs):
        M = inputs[0]
        u = inputs[1]
        output = T.batched_dot(M, u)
        if self.nonlinearity is not None:
            shape = output.shape
            output = self.nonlinearity(output.reshape((shape[0], -1))).reshape(shape)
        return output

class BatchedDotLayer(MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(BatchedDotLayer, self).__init__(incomings, **kwargs)
        if len(incomings) != 2:
            raise NotImplementedError

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][2])

    def get_output_for(self, inputs, **kwargs):
        return T.batched_dot(inputs[0], inputs[1])

class Model:
    def __init__(self, train_file, test_file, batch_size=50, embedding_size=20):
        self.train_lines, self.test_lines = self.get_lines(train_file), self.get_lines(test_file)
        lines = np.concatenate([np.array([{'type':'s', 'text':''}]), self.train_lines, self.test_lines], axis=0)

        self.vectorizer = CountVectorizer(lowercase=False)
        self.vectorizer.fit([x['text'] + ' ' + x['answer'] if 'answer' in x else x['text'] for x in lines])

        X = self.vectorizer.transform([x['text'] for x in lines]).toarray().astype(np.float32)

        self.data = { 'train': {}, 'test': {} }
        self.data['train']['C'], self.data['train']['Q'], self.data['train']['Y'], train_seqlen = self.get_dataset(self.train_lines, X[:len(self.train_lines)+1], offset=0)
        self.data['test']['C'], self.data['test']['Q'], self.data['test']['Y'], test_seqlen = self.get_dataset(self.test_lines, X[len(self.test_lines)+1:], offset=len(self.train_lines)+1)
        max_seqlen = max(train_seqlen, test_seqlen)

        self.batch_size = batch_size
        self.max_seqlen = max_seqlen
        self.num_classes = len(self.vectorizer.vocabulary_)

        c = T.imatrix()
        q = T.ivector()
        y = T.imatrix()
        self.c_shared = theano.shared(np.zeros((batch_size, max_seqlen), dtype=np.int32), borrow=True)
        self.q_shared = theano.shared(np.zeros((batch_size,), dtype=np.int32), borrow=True)
        self.a_shared = theano.shared(np.zeros((batch_size, self.num_classes), dtype=np.int32), borrow=True)

        l_C_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen))
        l_C_in = lasagne.layers.ReshapeLayer(l_C_in, shape=(batch_size * max_seqlen,))

        l_C_vec = lasagne.layers.EmbeddingLayer(l_C_in, input_size=X.shape[0], output_size=X.shape[1], W=X)
        l_C_vec.params[l_C_vec.W].remove('trainable')

        l_C_vec = lasagne.layers.ReshapeLayer(l_C_vec, shape=(batch_size * max_seqlen, X.shape[1]))
        l_C_embedding = lasagne.layers.DenseLayer(l_C_vec, embedding_size, b=None, nonlinearity=None)
        l_C_embedding = lasagne.layers.ReshapeLayer(l_C_embedding, shape=(batch_size, max_seqlen, embedding_size))

        l_Q_in = lasagne.layers.InputLayer(shape=(batch_size,))

        l_Q_vec = lasagne.layers.EmbeddingLayer(l_Q_in, input_size=X.shape[0], output_size=X.shape[1], W=X)
        l_Q_vec.params[l_Q_vec.W].remove('trainable')

        l_Q_vec = lasagne.layers.ReshapeLayer(l_Q_vec, shape=(batch_size, 1 * X.shape[1]))

        l_Q_embedding = lasagne.layers.DenseLayer(l_Q_vec, embedding_size, b=None, nonlinearity=None)

        l_prob = InnerProductLayer((l_C_embedding, l_Q_embedding), nonlinearity=lasagne.nonlinearities.softmax)

        l_C_embedding = lasagne.layers.ReshapeLayer(l_C_embedding, shape=(batch_size * max_seqlen, embedding_size))
        l_output = lasagne.layers.DenseLayer(l_C_embedding, embedding_size, b=None, nonlinearity=None)
        l_output = lasagne.layers.ReshapeLayer(l_output, shape=(batch_size, max_seqlen, embedding_size))

        l_weighted_output = BatchedDotLayer((l_prob, l_output))

        l_sum = lasagne.layers.ElemwiseSumLayer((l_weighted_output, l_Q_embedding))
        l_pred = lasagne.layers.DenseLayer(l_sum, self.num_classes, nonlinearity=lasagne.nonlinearities.softmax)

        probas = lasagne.layers.helper.get_output(l_pred, { l_C_in: c, l_Q_in: q })
        pred = T.argmax(probas, axis=1)
        cost = T.nnet.binary_crossentropy(probas, y).mean()

        params = lasagne.layers.helper.get_all_params(l_pred)
        updates = lasagne.updates.adam(cost, params)
        givens = {
            c: self.c_shared,
            q: self.q_shared,
            y: self.a_shared
        }

        self.train_model = theano.function([], cost, givens=givens, updates=updates)
        self.compute_pred = theano.function([], pred, givens=givens, on_unused_input='ignore')

    def predict(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.compute_pred()

    def compute_f1(self, dataset):
        n_batches = len(dataset['Y']) // self.batch_size
        y_pred = np.concatenate([self.predict(dataset, i) for i in xrange(n_batches)])
        y_true = [self.vectorizer.vocabulary_[y] for y in dataset['Y'][:len(y_pred)]]
        return metrics.f1_score(y_true, y_pred, average='weighted')

    def train(self, n_epochs=10, shuffle_batch=False):
        epoch = 0
        n_train_batches = len(self.data['train']['Y']) // self.batch_size
        n_test_batches = len(self.data['test']['Y']) // self.batch_size

        while (epoch < n_epochs):
            epoch += 1
            indices = range(n_train_batches)
            if shuffle_batch:
                indices = np.random.permutation(indices)
            total_cost = 0
            start_time = time.time()
            for minibatch_index in indices:
                self.set_shared_variables(self.data['train'], minibatch_index)
                total_cost += self.train_model()
            end_time = time.time()
            print "cost: ", (total_cost / len(indices)), " took: %d(s)" % (end_time - start_time)
            train_f1 = self.compute_f1(self.data['train'])
            print 'epoch %i, train_f1 %.2f' % (epoch, train_f1*100)

        test_f1 = self.compute_f1(self.data['test'])
        print 'test_f1: %.2f' % (test_f1*100)
        return test_f1

    def set_shared_variables(self, dataset, index):
        c = np.zeros((self.batch_size, self.max_seqlen), dtype=np.int32)
        q = np.zeros((self.batch_size,), dtype=np.int32)
        y = np.zeros((self.batch_size, self.num_classes), dtype=np.int32)

        indices = range(index*self.batch_size, (index+1)*self.batch_size)
        c_data = dataset['C'][indices]
        for i,row in enumerate(c_data):
            row = row[:self.max_seqlen]
            c[i,:len(row)] = row
        q[:len(indices)] = dataset['Q'][indices]
        y[:len(indices)] = self.vectorizer.transform(dataset['Y'][indices]).toarray()

        self.c_shared.set_value(c)
        self.q_shared.set_value(q)
        self.a_shared.set_value(y)

    def get_dataset(self, lines, vectorized_lines, offset):
        C, Q, Y = [], [], []
        max_seqlen = 0
        for i,line in enumerate(lines):
            if line['type'] == 'q':
                id = line['id']-1
                indices = [offset+idx for idx in range(i-id, i) if lines[idx]['type'] == 's']
                max_seqlen = max(len(indices), max_seqlen)
                C.append(indices)
                Q.append(offset+i)
                Y.append(line['answer'])
        return np.array(C), np.array(Q, dtype=np.int32), np.array(Y), max_seqlen

    def get_lines(self, fname):
        lines = []
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
    print "*" * 80
    print "args: ", args
    print "*" * 80

    train_file = glob.glob('data/en-10k/qa%d_*train.txt' % args.task)[0]
    test_file = glob.glob('data/en-10k/qa%d_*test.txt' % args.task)[0]
    if args.train_file != '' and args.test_file != '':
        train_file, test_file = args.train_file, args.test_file

    model = Model(train_file, test_file)
    model.train()

if __name__ == '__main__':
    main()
