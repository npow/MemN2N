from __future__ import division
import argparse
import cPickle
import glob
import lasagne
import nltk
import numpy as np
import pyprind
import sys
import theano
import theano.tensor as T
import time
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
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
        return T.sum(input, axis=self.axis)


class TemporalEncodingLayer(lasagne.layers.Layer):

    def __init__(self, incoming, T=lasagne.init.Normal(std=0.1), **kwargs):
        super(TemporalEncodingLayer, self).__init__(incoming, **kwargs)
        self.T = self.add_param(T, self.input_shape[-2:], name="T")

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return input + self.T


class TransposedDenseLayer(lasagne.layers.DenseLayer):

    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
                 **kwargs):
        super(TransposedDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)

        activation = T.dot(input, self.W.T)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


class MemoryNetworkLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, vocab, embedding_size, A, A_T, C, C_T, nonlinearity=lasagne.nonlinearities.softmax, **kwargs):
        super(MemoryNetworkLayer, self).__init__(incomings, **kwargs)
        if len(incomings) != 3:
            raise NotImplementedError

        batch_size, max_seqlen, max_sentlen = self.input_shapes[0]

        l_context_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen))
        l_B_embedding = lasagne.layers.InputLayer(shape=(batch_size, embedding_size))
        l_context_pe_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen, embedding_size))

        l_context_in = lasagne.layers.ReshapeLayer(l_context_in, shape=(batch_size * max_seqlen * max_sentlen, ))
        l_A_embedding = lasagne.layers.EmbeddingLayer(l_context_in, len(vocab)+1, embedding_size, W=A)
        self.A = l_A_embedding.W
        l_A_embedding = lasagne.layers.ReshapeLayer(l_A_embedding, shape=(batch_size, max_seqlen, max_sentlen, embedding_size))
        l_A_embedding = lasagne.layers.ElemwiseMergeLayer((l_A_embedding, l_context_pe_in), merge_function=T.mul)
        l_A_embedding = SumLayer(l_A_embedding, axis=2)
        l_A_embedding = TemporalEncodingLayer(l_A_embedding, T=A_T)
        self.A_T = l_A_embedding.T

        l_C_embedding = lasagne.layers.EmbeddingLayer(l_context_in, len(vocab)+1, embedding_size, W=C)
        self.C = l_C_embedding.W
        l_C_embedding = lasagne.layers.ReshapeLayer(l_C_embedding, shape=(batch_size, max_seqlen, max_sentlen, embedding_size))
        l_C_embedding = lasagne.layers.ElemwiseMergeLayer((l_C_embedding, l_context_pe_in), merge_function=T.mul)
        l_C_embedding = SumLayer(l_C_embedding, axis=2)
        l_C_embedding = TemporalEncodingLayer(l_C_embedding, T=C_T)
        self.C_T = l_C_embedding.T

        l_prob = InnerProductLayer((l_A_embedding, l_B_embedding), nonlinearity=nonlinearity)
        l_weighted_output = BatchedDotLayer((l_prob, l_C_embedding))

        l_sum = lasagne.layers.ElemwiseSumLayer((l_weighted_output, l_B_embedding))

        self.l_context_in = l_context_in
        self.l_B_embedding = l_B_embedding
        self.l_context_pe_in = l_context_pe_in
        self.network = l_sum

        params = lasagne.layers.helper.get_all_params(self.network, trainable=True)
        values = lasagne.layers.helper.get_all_param_values(self.network, trainable=True)
        for p, v in zip(params, values):
            self.add_param(p, v.shape, name=p.name)

        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [self.A, self.C]])

    def get_output_shape_for(self, input_shapes):
        return lasagne.layers.helper.get_output_shape(self.network)

    def get_output_for(self, inputs, **kwargs):
        return lasagne.layers.helper.get_output(self.network, {self.l_context_in: inputs[0], self.l_B_embedding: inputs[1], self.l_context_pe_in: inputs[2]})

    def reset_zero(self):
        self.set_zero(self.zero_vec)


class Model:

    def __init__(self,
                 data,
                 vocab,
                 S,
                 batch_size=32,
                 embedding_size=20,
                 max_norm=40,
                 lr=0.01,
                 num_hops=3,
                 adj_weight_tying=False,
                 linear_start=True,
                 input_dir='dataset_1MM',
                 suffix='',
                 max_seqlen=20,
                 max_sentlen=20,
                 **kwargs):

        self.data = data
        self.vocab = vocab
        self.S = S
        self.num_classes = 1

        print 'batch_size:', batch_size, 'max_seqlen:', max_seqlen, 'max_sentlen:', max_sentlen
        for d in ['train', 'test']:
            print d,
            for k in ['C', 'Q', 'Y']:
                print k, self.data[d][k].shape,
            print ''

        self.batch_size = batch_size
        self.max_seqlen = max_seqlen
        self.max_sentlen = max_sentlen
        self.embedding_size = embedding_size
        self.adj_weight_tying = adj_weight_tying
        self.num_hops = num_hops
        self.init_lr = lr
        self.lr = self.init_lr
        self.max_norm = max_norm
        self.nonlinearity = None if linear_start else lasagne.nonlinearities.softmax

        self.build_network(self.nonlinearity)

    def build_network(self, nonlinearity):
        batch_size, max_seqlen, max_sentlen, embedding_size, vocab = self.batch_size, self.max_seqlen, self.max_sentlen, self.embedding_size, self.vocab

        c = T.imatrix()
        q = T.ivector()
        y = T.imatrix()
        c_pe = T.tensor4()
        q_pe = T.tensor4()
        self.c_shared = theano.shared(np.zeros((batch_size, max_seqlen), dtype=np.int32), borrow=True)
        self.q_shared = theano.shared(np.zeros((batch_size, ), dtype=np.int32), borrow=True)
        self.a_shared = theano.shared(np.zeros((batch_size, self.num_classes), dtype=np.int32), borrow=True)
        self.c_pe_shared = theano.shared(np.zeros((batch_size, max_seqlen, max_sentlen, embedding_size), dtype=theano.config.floatX), borrow=True)
        self.q_pe_shared = theano.shared(np.zeros((batch_size, 1, max_sentlen, embedding_size), dtype=theano.config.floatX), borrow=True)
        S_shared = theano.shared(self.S, borrow=True)
        
        cc = S_shared[c.flatten()].reshape((batch_size, max_seqlen, max_sentlen))
        qq = S_shared[q.flatten()].reshape((batch_size, max_sentlen))

        l_context_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen))
        l_question_in = lasagne.layers.InputLayer(shape=(batch_size, max_sentlen))

        l_context_pe_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen, embedding_size))
        l_question_pe_in = lasagne.layers.InputLayer(shape=(batch_size, 1, max_sentlen, embedding_size))

        A, C = lasagne.init.Normal(std=0.1).sample((len(vocab)+1, embedding_size)), lasagne.init.Normal(std=0.1)
        A_T, C_T = lasagne.init.Normal(std=0.1), lasagne.init.Normal(std=0.1)
        W = A if self.adj_weight_tying else lasagne.init.Normal(std=0.1)

        l_question_in = lasagne.layers.ReshapeLayer(l_question_in, shape=(batch_size * max_sentlen, ))
        l_B_embedding = lasagne.layers.EmbeddingLayer(l_question_in, len(vocab)+1, embedding_size, W=W)
        B = l_B_embedding.W
        l_B_embedding = lasagne.layers.ReshapeLayer(l_B_embedding, shape=(batch_size, 1, max_sentlen, embedding_size))
        l_B_embedding = lasagne.layers.ElemwiseMergeLayer((l_B_embedding, l_question_pe_in), merge_function=T.mul)
        l_B_embedding = lasagne.layers.ReshapeLayer(l_B_embedding, shape=(batch_size, max_sentlen, embedding_size))
        l_B_embedding = SumLayer(l_B_embedding, axis=1)

        self.mem_layers = [MemoryNetworkLayer((l_context_in, l_B_embedding, l_context_pe_in), vocab, embedding_size, A=A, A_T=A_T, C=C, C_T=C_T, nonlinearity=nonlinearity)]
        for _ in range(1, self.num_hops):
            if self.adj_weight_tying:
                A, C = self.mem_layers[-1].C, lasagne.init.Normal(std=0.1)
                A_T, C_T = self.mem_layers[-1].C_T, lasagne.init.Normal(std=0.1)
            else:  # RNN style
                A, C = self.mem_layers[-1].A, self.mem_layers[-1].C
                A_T, C_T = self.mem_layers[-1].A_T, self.mem_layers[-1].C_T
            self.mem_layers += [MemoryNetworkLayer((l_context_in, self.mem_layers[-1], l_context_pe_in), vocab, embedding_size, A=A, A_T=A_T, C=C, C_T=C_T, nonlinearity=nonlinearity)]

        if False and self.adj_weight_tying:
            l_pred = TransposedDenseLayer(self.mem_layers[-1], self.num_classes, W=self.mem_layers[-1].C, b=None, nonlinearity=lasagne.nonlinearities.softmax)
        else:
            l_pred = lasagne.layers.DenseLayer(self.mem_layers[-1], self.num_classes, W=lasagne.init.Normal(std=0.1), b=None, nonlinearity=lasagne.nonlinearities.softmax)

        o = lasagne.layers.helper.get_output(l_pred, {l_context_in: cc, l_question_in: qq, l_context_pe_in: c_pe, l_question_pe_in: q_pe})
        o = T.clip(o, 1e-7, 1.0-1e-7)

        probas = T.concatenate([(1-o).reshape((-1,1)), o.reshape((-1,1))], axis=1)
        pred = T.argmax(probas, axis=1, keepdims=True)
        errors = T.sum(T.neq(pred, y))

        cost = T.nnet.binary_crossentropy(o, y).sum()

        params = lasagne.layers.helper.get_all_params(l_pred, trainable=True)
        print 'params:', params
        grads = T.grad(cost, params)
        scaled_grads = lasagne.updates.total_norm_constraint(grads, self.max_norm)
        updates = lasagne.updates.sgd(scaled_grads, params, learning_rate=self.lr)

        givens = {
            c: self.c_shared,
            q: self.q_shared,
            y: self.a_shared,
            c_pe: self.c_pe_shared,
            q_pe: self.q_pe_shared
        }

        self.train_model = theano.function([], cost, givens=givens, updates=updates)
        self.get_probas = theano.function([], probas, givens=givens, on_unused_input='warn')
        self.get_loss = theano.function([], errors, givens=givens, on_unused_input='warn')

        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [B]])

        self.nonlinearity = nonlinearity
        self.network = l_pred

    def reset_zero(self):
        self.set_zero(self.zero_vec)
        for l in self.mem_layers:
            l.reset_zero()

    def compute_loss(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.get_loss()

    def compute_probas(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.get_probas()[:,1]

    def report_perf(self, dataset):
        n_batches = len(dataset['Y']) // self.batch_size
        probas = np.concatenate([self.compute_probas(dataset, i) for i in xrange(n_batches)])
        self.compute_recall_ks(probas)

    def compute_recall_ks(self, probas):
      recall_k = {}
      for group_size in [2, 10]:
          recall_k[group_size] = {}
          print 'group_size: %d' % group_size
          for k in [1, 2, 5]:
              if k < group_size:
                  recall_k[group_size][k] = self.recall(probas, k, group_size)
                  print 'recall@%d' % k, recall_k[group_size][k]
      return recall_k
                
    def recall(self, probas, k, group_size):
        test_size = 10
        n_batches = len(probas) // test_size
        n_correct = 0
        for i in xrange(n_batches):
            batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
            #p = np.random.permutation(len(batch))
            #indices = p[np.argpartition(batch[p], -k)[-k:]]
            indices = np.argpartition(batch, -k)[-k:]
            if 0 in indices:
                n_correct += 1
        return n_correct / (len(probas) / test_size)

    def train(self, n_epochs=100, shuffle_batch=False):
        epoch = 0
        best_val_perf = 0
        best_val_rk1 = 0
        prev_val_rk1 = None
        test_perf = 0
        cost_epoch = 0

        n_train_batches = len(self.data['train']['Y']) // self.batch_size
        n_val_batches = len(self.data['val']['Y']) // self.batch_size
        n_test_batches = len(self.data['test']['Y']) // self.batch_size

        while (epoch < n_epochs):
            epoch += 1

            if epoch % 25 == 0:
                self.lr /= 2.0

            indices = range(n_train_batches)
            if shuffle_batch:
                self.shuffle_sync(self.data['train'])
            bar = pyprind.ProgBar(len(indices), monitor=True)
            total_cost = 0
            start_time = time.time()
            for minibatch_index in indices:
                self.set_shared_variables(self.data['train'], minibatch_index)
                cost_epoch = self.train_model()
                total_cost += cost_epoch
                self.reset_zero()
                bar.update()
            end_time = time.time()
            print "cost: ", (total_cost / len(indices)), " took: %d(s)" % (end_time - start_time)
            train_losses = [self.compute_loss(self.data['train'], i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.sum(train_losses) / len(self.data['train']['Y'])
            val_losses = [self.compute_loss(self.data['val'], i) for i in xrange(n_val_batches)]
            val_perf = 1 - np.sum(val_losses) / len(self.data['val']['Y'])
            print 'epoch %i, train_perf %f, val_perf %f' % (epoch, train_perf*100, val_perf*100)

            val_probas = np.concatenate([self.compute_probas(self.data['val'], i) for i in xrange(n_val_batches)])
            val_recall_k = self.compute_recall_ks(val_probas)
            val_rk1 = val_recall_k[10][1]

            if prev_val_rk1 is not None and val_rk1 > prev_val_rk1 and self.nonlinearity is None:
                prev_weights = lasagne.layers.helper.get_all_param_values(self.network)
                self.build_network(nonlinearity=lasagne.nonlinearities.softmax)
                lasagne.layers.helper.set_all_param_values(self.network, prev_weights)
            else:
                if val_perf > best_val_perf or val_rk1 > best_val_rk1:
                    best_val_perf = val_perf
                    best_val_rk1 = val_rk1
                    test_losses = [self.compute_loss(self.data['test'], i) for i in xrange(n_test_batches)]
                    test_perf = 1 - np.sum(test_losses) / len(self.data['test']['Y'])
                    print 'test_perf: %f' % (test_perf*100)
                    test_probas = np.concatenate([self.compute_probas(self.data['test'], i) for i in xrange(n_test_batches)])
                    self.compute_recall_ks(test_probas)
                else:
                    break

            prev_val_rk1 = val_rk1

        return test_perf

    def shuffle_sync(self, dataset):
        p = np.random.permutation(len(dataset['Y']))
        for k in ['C', 'Q', 'Y']:
            dataset[k] = dataset[k][p]
            
    def set_shared_variables(self, dataset, index):
        c = np.zeros((self.batch_size, self.max_seqlen), dtype=np.int32)
        q = np.zeros((self.batch_size, ), dtype=np.int32)
        y = np.zeros((self.batch_size, self.num_classes), dtype=np.int32)
        c_pe = np.zeros((self.batch_size, self.max_seqlen, self.max_sentlen, self.embedding_size), dtype=theano.config.floatX)
        q_pe = np.zeros((self.batch_size, 1, self.max_sentlen, self.embedding_size), dtype=theano.config.floatX)

        indices = range(index*self.batch_size, (index+1)*self.batch_size)
        for i, row in enumerate(dataset['C'][indices]):
            row = row[:self.max_seqlen]
            c[i, :len(row)] = row

        q[:len(indices)] = dataset['Q'][indices]

        for key, mask in [('C', c_pe), ('Q', q_pe)]:
            for i, row in enumerate(dataset[key][indices]):
                sentences = self.S[row].reshape((-1, self.max_sentlen))
                for ii, word_idxs in enumerate(sentences):
                    J = np.count_nonzero(word_idxs)
                    for j in np.arange(J):
                        mask[i, ii, j, :] = (1 - (j+1)/J) - ((np.arange(self.embedding_size)+1)/self.embedding_size)*(1 - 2*(j+1)/J)

        y[:len(indices)] = dataset['Y'][indices].reshape((-1, 1))

        self.c_shared.set_value(c)
        self.q_shared.set_value(q)
        self.a_shared.set_value(y)
        self.c_pe_shared.set_value(c_pe)
        self.q_pe_shared.set_value(q_pe)


def split_utterances(utterances, eos_idx):
    return [[int(yy) for yy in y.split()] for y in ' '.join([str(x) for x in utterances]).split(eos_idx)]


def process_dataset(dataset, max_seqlen, max_sentlen, offset, eos_idx='63346'):
    S, C, Q, Y = [], [], [], []

    for i, row in enumerate(dataset['c']):
        utterances = split_utterances(row, eos_idx)
        for ii, utterance in enumerate(utterances):
            word_indices = utterance[:max_sentlen]
            word_indices += [0] * (max_sentlen - len(word_indices))
            S.append(word_indices)

            if ii == len(utterances)-1:
                indices = [offset+idx+1 for idx in range(len(S)-len(utterances)-1, len(S)-1)][:max_seqlen]
                C.append(indices)
                
                response = split_utterances(dataset['r'][i], eos_idx)[0][:max_sentlen]
                response += [0] * (max_sentlen - len(response))
                S.append(response)
                Q.append(offset+len(S)-1)
                
                Y.append(dataset['y'][i])
    return S, np.array(C), np.array(Q, dtype=np.int32), np.array(Y, dtype=np.int32)


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


def load_data(input_dir, suffix, max_seqlen, max_sentlen):
    if True:
        train_data, val_data, test_data = cPickle.load(open('%s/dataset%s.pkl' % (input_dir, suffix), 'rb'))
        vocab = cPickle.load(open('%s/vocab%s.pkl' % (input_dir, suffix), 'rb'))
        data = {'train': {}, 'val': {}, 'test': {}}
        S_train, data['train']['C'], data['train']['Q'], data['train']['Y'] = process_dataset(train_data, max_seqlen, max_sentlen, offset=0)
        S_val, data['val']['C'], data['val']['Q'], data['val']['Y'] = process_dataset(val_data, max_seqlen, max_sentlen, offset=len(S_train))
        S_test, data['test']['C'], data['test']['Q'], data['test']['Y'] = process_dataset(test_data, max_seqlen, max_sentlen, offset=len(S_train)+len(S_val))
        S = np.concatenate([np.zeros((1, max_sentlen), dtype=np.int32), S_train, S_val, S_test], axis=0)
    else:
        data, vocab, S = cPickle.load(open('blobs.pkl'))

    return data, vocab, S

def main():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--task', type=int, default=1, help='Task#')
    parser.add_argument('--train_file', type=str, default='', help='Train file')
    parser.add_argument('--test_file', type=str, default='', help='Test file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--embedding_size', type=int, default=20, help='Embedding size')
    parser.add_argument('--max_norm', type=float, default=40.0, help='Max norm')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_hops', type=int, default=3, help='Num hops')
    parser.add_argument('--adj_weight_tying', type='bool', default=True, help='Whether to use adjacent weight tying')
    parser.add_argument('--linear_start', type='bool', default=False, help='Whether to start with linear activations')
    parser.add_argument('--shuffle_batch', type='bool', default=False, help='Whether to shuffle minibatches')
    parser.add_argument('--n_epochs', type=int, default=100, help='Num epochs')
    parser.add_argument('--input_dir', type=str, default='dataset_1MM', help='Input dir')
    parser.add_argument('--suffix', type=str, default='', help='Suffix')
    parser.add_argument('--max_seqlen', type=int, default=20, help='Max seqlen')
    parser.add_argument('--max_sentlen', type=int, default=50, help='Max sentlen')
    args = parser.parse_args()
    print '*' * 80
    print 'args:', args
    print '*' * 80

    args.data, args.vocab, args.S = load_data(args.input_dir, args.suffix, args.max_seqlen, args.max_sentlen)

    model = Model(**args.__dict__)
    model.train(n_epochs=args.n_epochs, shuffle_batch=args.shuffle_batch)

if __name__ == '__main__':
    main()
