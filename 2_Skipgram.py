# -*- coding: utf-8 -*-
from dataset import *
import numpy as np
import math
import tensorflow as tf
import pickle
from const import *


class Skipgram(object):
    def __init__(self, corpus):
        self.corpus = corpus
        self.V = corpus.V
        self.N = 100
        self.BATCH_SIZE = 128
        self.WIN_SIZE = 4
        self.BATCH_SIZE *= self.WIN_SIZE    # to generate data with same num as cbow
        self.g = corpus.generate_batch_skipgram(self.BATCH_SIZE, self.WIN_SIZE)
        self.valid_size = 5
        self.valid_window = 5
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        self.num_sampled = 64

    def train(self):
        with tf.Graph().as_default() as graph:
            train_inputs = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE])
            train_labels = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE, 1])
            embeddings = tf.Variable(tf.random_uniform(shape=[self.V, self.N], minval=-1.0, maxval=1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            #nce_bias = tf.Variable(tf.zeros([self.V]))
            nce_weights = tf.Variable(tf.truncated_normal([self.V, self.N], stddev=1.0 / math.sqrt(self.N)))
            nce_bias = tf.constant(0, dtype=np.float32, shape=[self.V])

            loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_bias, labels=train_labels,
                                                 inputs=embed, num_sampled=self.num_sampled, num_classes=self.V))
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            normalized_embeddings = embeddings / tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

            init = tf.global_variables_initializer()


        with tf.Session(graph=graph) as session:
            init.run()
            print("initialized")
            average_loss = 0.0
            for step in range(TRAIN_STEPS):
                batch_inputs, batch_labels = next(self.g)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val
                if (step + 1) % PRINT_FREQ == 0:
                    average_loss /= PRINT_FREQ
                    print("Average loss at step", step, ":", average_loss)
                    average_loss = 0
                if (step + 1) % TEST_FREQ == 0:
                    sim = similarity.eval()     # type <class 'numpy.ndarray'>
                    for i in range(self.valid_size):    # sim.shape[0]
                        valid_word = self.corpus.idx2word[self.valid_examples[i]]
                        top_k = 8
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in range(top_k):
                            close_word = self.corpus.idx2word[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)
            final_embeddings = normalized_embeddings.eval()
            with open(RESULT_DIR+'final_embeddings_skipgram.pkl', 'wb') as f:
                pickle.dump([final_embeddings, self.corpus.idx2word], f)


if __name__ == '__main__':
    file_name = download_data(URL, EXPECTED_BYTES, LOCAL_DATA_DIR)
    word_list = get_word_list(file_name)
    corpus = Corpus(word_list)
    net = Skipgram(corpus)
    net.train()
