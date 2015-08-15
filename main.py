from __future__ import division
import argparse
import glob
import lasagne
import numpy as np
import sys
import theano
import theano.tensor as T
import time
from sklearn import metrics
from sklearn.feature_extraction.text import *
from sklearn.preprocessing import *
from theano.printing import Print as pp

import warnings
warnings.filterwarnings('ignore', '.*topo.*')

class InnerProductLayer(lasagne.layers.MergeLayer):
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
            output = self.nonlinearity(output)
        return output

class BatchedDotLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(BatchedDotLayer, self).__init__(incomings, **kwargs)
        if len(incomings) != 2:
            raise NotImplementedError

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][2])

    def get_output_for(self, inputs, **kwargs):
        return T.batched_dot(inputs[0], inputs[1])

class SumLayer(lasagne.layers.Layer):
    def __init__(self, incoming, axis, **kwargs):
        super(SumLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        return input_shape[:self.axis] + input_shape[self.axis+1:]

    def get_output_for(self, input, **kwargs):
        return T.sum(input, axis=self.axis, dtype=theano.config.floatX)

class Model:
    def __init__(self, train_file, test_file, batch_size=32, embedding_size=20, max_norm=40, lr=0.01):
        train_lines, test_lines = self.get_lines(train_file), self.get_lines(test_file)
        lines = np.concatenate([train_lines, test_lines], axis=0)
        vocab, word_to_idx, max_seqlen, max_sentlen = self.get_vocab(lines)

        self.data = { 'train': {}, 'test': {} }
        S_train, self.data['train']['C'], self.data['train']['Q'], self.data['train']['Y'] = self.process_dataset(train_lines, word_to_idx, max_sentlen)
        S_test, self.data['test']['C'], self.data['test']['Q'], self.data['test']['Y'] = self.process_dataset(test_lines, word_to_idx, max_sentlen)
        S = np.concatenate([S_train, S_test], axis=0)

        print 'batch_size:', batch_size, 'max_seqlen:', max_seqlen, 'max_sentlen:', max_sentlen
        print 'sentences:', S.shape
        for d in ['train', 'test']:
            print d,
            for k in ['C', 'Q', 'Y']:
                print k, self.data[d][k].shape,
            print ''

        lb = LabelBinarizer()
        lb.fit(list(vocab))
        vocab = lb.classes_.tolist()

        self.batch_size = batch_size
        self.max_seqlen = max_seqlen
        self.max_sentlen = max_sentlen
        self.num_classes = len(vocab)
        self.vocab = vocab
        self.lb = lb
        self.init_lr = lr
        self.lr = self.init_lr

        c = T.imatrix()
        q = T.ivector()
        y = T.imatrix()
        self.c_shared = theano.shared(np.zeros((batch_size, max_seqlen), dtype=np.int32), borrow=True)
        self.q_shared = theano.shared(np.zeros((batch_size, ), dtype=np.int32), borrow=True)
        self.a_shared = theano.shared(np.zeros((batch_size, self.num_classes), dtype=np.int32), borrow=True)
        S_shared = theano.shared(S, borrow=True)

        cc = S_shared[c.flatten()].reshape((batch_size, max_seqlen, max_sentlen))
        qq = S_shared[q.flatten()].reshape((batch_size, max_sentlen))

        l_C_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen))
        l_C_in = lasagne.layers.ReshapeLayer(l_C_in, shape=(batch_size * max_seqlen * max_sentlen, ))
        l_C_embedding = lasagne.layers.EmbeddingLayer(l_C_in, len(vocab)+1, embedding_size)
        A = l_C_embedding.W
        l_C_embedding = lasagne.layers.ReshapeLayer(l_C_embedding, shape=(batch_size, max_seqlen, max_sentlen, embedding_size))
        l_C_embedding = SumLayer(l_C_embedding, axis=2)

        l_Q_in = lasagne.layers.InputLayer(shape=(batch_size, max_sentlen))
        l_Q_in = lasagne.layers.ReshapeLayer(l_Q_in, shape=(batch_size * max_sentlen, ))
        l_Q_embedding = lasagne.layers.EmbeddingLayer(l_Q_in, len(vocab)+1, embedding_size)
        B = l_Q_embedding.W
        l_Q_embedding = lasagne.layers.ReshapeLayer(l_Q_embedding, shape=(batch_size, max_sentlen, embedding_size))
        l_Q_embedding = SumLayer(l_Q_embedding, axis=1)

        l_prob = InnerProductLayer((l_C_embedding, l_Q_embedding), nonlinearity=lasagne.nonlinearities.softmax)

        l_C_embedding = lasagne.layers.ReshapeLayer(l_C_embedding, shape=(batch_size * max_seqlen, embedding_size))
        l_output = lasagne.layers.DenseLayer(l_C_embedding, embedding_size, W=lasagne.init.Normal(std=0.1), b=lasagne.init.Constant(0), nonlinearity=None)
        l_output = lasagne.layers.ReshapeLayer(l_output, shape=(batch_size, max_seqlen, embedding_size))

        l_weighted_output = BatchedDotLayer((l_prob, l_output))

        l_sum = lasagne.layers.ElemwiseSumLayer((l_weighted_output, l_Q_embedding))
        l_pred = lasagne.layers.DenseLayer(l_sum, self.num_classes, W=lasagne.init.Normal(std=0.1), b=lasagne.init.Constant(0), nonlinearity=lasagne.nonlinearities.softmax)

        probas = lasagne.layers.helper.get_output(l_pred, { l_C_in: cc, l_Q_in: qq })
        probas = T.clip(probas, 1e-7, 1.0-1e-7)

        pred = T.argmax(probas, axis=1)

        cost = T.nnet.binary_crossentropy(probas, y).sum()

        params = lasagne.layers.helper.get_all_params(l_pred, trainable=True)
        grads = T.grad(cost, params)
        scaled_grads = lasagne.updates.total_norm_constraint(grads, max_norm)
        updates = lasagne.updates.adam(scaled_grads, params, learning_rate=self.lr)

        givens = {
            c: self.c_shared,
            q: self.q_shared,
            y: self.a_shared
        }

        self.train_model = theano.function([], cost, givens=givens, updates=updates)
        self.compute_pred = theano.function([], pred, givens=givens, on_unused_input='ignore')

        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(A, T.set_subtensor(A[0,:], zero_vec_tensor)), (B, T.set_subtensor(B[0,:], zero_vec_tensor))])
        self.set_zero(self.zero_vec)

    def predict(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.compute_pred()

    def compute_f1(self, dataset):
        n_batches = len(dataset['Y']) // self.batch_size
        y_pred = np.concatenate([self.predict(dataset, i) for i in xrange(n_batches)])
        y_true = [self.vocab.index(y) for y in dataset['Y'][:len(y_pred)]]
        print metrics.confusion_matrix(y_true, y_pred)
        print metrics.classification_report(y_true, y_pred)
        return metrics.f1_score(y_true, y_pred, average='weighted')

    def train(self, n_epochs=100, shuffle_batch=False):
        epoch = 0
        n_train_batches = len(self.data['train']['Y']) // self.batch_size
        n_test_batches = len(self.data['test']['Y']) // self.batch_size
        self.lr = self.init_lr

        while (epoch < n_epochs):
            epoch += 1

            if epoch % 25 == 0:
                self.lr /= 2.0

            indices = range(n_train_batches)
            if shuffle_batch:
                self.shuffle_sync(self.data['train'])

            total_cost = 0
            start_time = time.time()
            for minibatch_index in indices:
                self.set_shared_variables(self.data['train'], minibatch_index)
                total_cost += self.train_model()
                self.set_zero(self.zero_vec)
            end_time = time.time()
            print '\n' * 3, '*' * 80
            print 'epoch:', epoch, 'cost:', (total_cost / len(indices)), ' took: %d(s)' % (end_time - start_time)

            print 'TRAIN', '=' * 40
            train_f1 = self.compute_f1(self.data['train'])

            print 'TEST', '=' * 40
            test_f1 = self.compute_f1(self.data['test'])

    def shuffle_sync(self, dataset):
        p = np.random.permutation(len(dataset['Y']))
        for k in ['C', 'Q', 'Y']:
            dataset[k] = dataset[k][p]

    def set_shared_variables(self, dataset, index):
        c = np.zeros((self.batch_size, self.max_seqlen), dtype=np.int32)
        q = np.zeros((self.batch_size, ), dtype=np.int32)
        y = np.zeros((self.batch_size, self.num_classes), dtype=np.int32)

        indices = range(index*self.batch_size, (index+1)*self.batch_size)
        for i,row in enumerate(dataset['C'][indices]):
            row = row[:self.max_seqlen]
            c[i,:len(row)] = row
        q[:len(indices)] = dataset['Q'][indices]
        y[:len(indices)] = self.lb.transform(dataset['Y'][indices])

        self.c_shared.set_value(c)
        self.q_shared.set_value(q)
        self.a_shared.set_value(y)

    def get_vocab(self, lines):
        vocab = set()
        max_sentlen = 0
        for i,line in enumerate(lines):
            words = line['text'].split()
            max_sentlen = max(max_sentlen, len(words))
            for w in words:
                vocab.add(w)
            if line['type'] == 'q':
                vocab.add(line['answer'])

        word_to_idx = {}
        for w in vocab:
            word_to_idx[w] = len(word_to_idx) + 1

        max_seqlen = 0
        for i,line in enumerate(lines):
            if line['type'] == 'q':
                id = line['id']-1
                indices = [idx for idx in range(i-id, i) if lines[idx]['type'] == 's']
                max_seqlen = max(len(indices), max_seqlen)

        return vocab, word_to_idx, max_seqlen, max_sentlen

    def process_dataset(self, lines, word_to_idx, max_sentlen):
        S, C, Q, Y = [], [], [], []

        for i,line in enumerate(lines):
            word_indices = [word_to_idx[w] for w in line['text'].split()]
            word_indices += [0] * (max_sentlen - len(word_indices))
            S.append(word_indices)
            if line['type'] == 'q':
                id = line['id']-1
                indices = [idx for idx in range(i-id, i) if lines[idx]['type'] == 's']
                C.append(indices)
                Q.append(i)
                Y.append(line['answer'])
        return np.array(S, dtype=np.int32), np.array(C), np.array(Q, dtype=np.int32), np.array(Y)

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
  return v.lower() in ('yes', 'true', 't', '1')

def main():
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.add_argument('--task', type=int, default=1, help='Task#')
    parser.add_argument('--train_file', type=str, default='', help='Train file')
    parser.add_argument('--test_file', type=str, default='', help='Test file')
    args = parser.parse_args()
    print '*' * 80
    print 'args:', args
    print '*' * 80

    train_file = glob.glob('data/en-10k/qa%d_*train.txt' % args.task)[0]
    test_file = glob.glob('data/en-10k/qa%d_*test.txt' % args.task)[0]
    if args.train_file != '' and args.test_file != '':
        train_file, test_file = args.train_file, args.test_file

    model = Model(train_file, test_file)
    model.train(n_epochs=100, shuffle_batch=True)

if __name__ == '__main__':
    main()
