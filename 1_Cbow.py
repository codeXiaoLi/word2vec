# -*- coding: utf-8 -*-
from dataset import *
import numpy as np
import math
import tensorflow as tf
import pickle
from const import *


class Cbownet(object):
    def __init__(self, corpus):
        self.corpus = corpus
        self.V = corpus.V
        self.N = 100
        self.num_sampled = 30
        self.BATCH_SIZE = 256
        self.WIN_SIZE = 4
        self.g = corpus.generate_batch_cbow(self.BATCH_SIZE, self.WIN_SIZE)
        self.valid_size = 5
        self.valid_window = 5
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)

    def train(self):
        with tf.Graph().as_default() as graph:
            train_inputs = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE, self.WIN_SIZE])
            train_labels = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE, 1])
            embeddings = tf.Variable(tf.random_uniform(shape=[self.V, self.N], minval=-1.0, maxval=1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            embed = tf.reduce_sum(embed, 1)
            #nce_bias = tf.Variable(tf.zeros([self.V]))
            nce_weights = tf.Variable(tf.truncated_normal([self.V, self.N], stddev=1.0 / math.sqrt(self.N)))
            nce_bias = tf.constant(0, dtype=np.float32, shape=[self.V])

            loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_bias, labels=train_labels,
                                                 inputs=embed, num_sampled=self.num_sampled, num_classes=self.V))   # , num_true=3
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            normalized_embeddings = embeddings / tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

            eval_sem_dataset = tf.placeholder(tf.int32, shape=[None, 4])
            eval_sem_dataset_012 = eval_sem_dataset[:, 0:3]
            eval_sem_dataset_3 = eval_sem_dataset[:, 3]

            eval_sem_embeddings_012 = tf.nn.embedding_lookup(normalized_embeddings, eval_sem_dataset_012)
            eval_sem_embeddings_1sub0add2 = eval_sem_embeddings_012[:, 1, :] - eval_sem_embeddings_012[:, 0, :] + eval_sem_embeddings_012[:, 2, :]
            eval_sem_similarity = tf.matmul(eval_sem_embeddings_1sub0add2, normalized_embeddings, transpose_b=True)
            #eval_sem_embeddings_3 = tf.nn.embedding_lookup(normalized_embeddings, eval_sem_dataset_3)
            #eval_sem_similarity = tf.matmul(eval_sem_embeddings_3, normalized_embeddings, transpose_b=True)
            eval_sem_dataset_3_pred = tf.nn.top_k(eval_sem_similarity, 1)[1][:, 0]
            eval_sem_score = tf.reduce_sum(tf.to_int32(tf.equal(eval_sem_dataset_3, eval_sem_dataset_3_pred)))



            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

            init = tf.global_variables_initializer()


        with tf.Session(graph=graph) as session:
            init.run()
            print("initialized, vec dim", self.N, '#neg sample', self.num_sampled, 'batchsize', self.BATCH_SIZE, 'winsize', self.WIN_SIZE)
            average_loss = 0.0
            for step in range(TRAIN_STEPS):
                batch_inputs, batch_labels = next(self.g)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val
                if (step + 1) % PRINT_FREQ == 0:
                    average_loss /= PRINT_FREQ
                    print("Average loss at step", step+1, ":", average_loss)
                    average_loss = 0
                if (step + 1) % EVAL_FREQ == 0:
                    lines = open(LOCAL_DATA_DIR + 'semantic.txt', encoding='utf-8').readlines()
                    score_sem = 0
                    for i in range(5):
                        eval_sem = np.array([[self.corpus.word2idx[word] for word in line.strip().split()] for line in lines[i*100:(i+1)*100]])
                        score_sem += session.run(eval_sem_score, feed_dict={eval_sem_dataset: eval_sem})
                    lines = open(LOCAL_DATA_DIR + 'syntactic.txt', encoding='utf-8').readlines()
                    score_syn = 0
                    for i in range(89):
                        eval_syn = np.array([[self.corpus.word2idx[word] for word in line.strip().split()] for line in lines[i*100:(i+1)*100]])
                        score_syn += session.run(eval_sem_score, feed_dict={eval_sem_dataset: eval_syn})
                    #print("Evaluation at step", step+1)
                    #print('Semantic score:', score_sem)
                    #print('Semantic score:', score_syn)
                    print('Semantic score:', score_sem+score_syn)
                    '''
                    print('Semantic score:', score_sem/506)
                    print('Semantic score:', score_syn/8946)
                    print('Semantic score:', (score_sem+score_syn)/(506+8946))
                    
                    '''
                '''
                if (step + 1) % TEST_FREQ == 0:
                    sim = similarity.eval() # type <class 'numpy.ndarray'>
                    for i in range(self.valid_size):    # sim.shape[0]
                        valid_word = self.corpus.idx2word[self.valid_examples[i]]
                        top_k = 8
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in range(top_k):
                            close_word = self.corpus.idx2word[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)
                '''
            final_embeddings = normalized_embeddings.eval()
            with open(RESULT_DIR+'final_embeddings_cbow.pkl', 'wb') as f:
                pickle.dump([final_embeddings, self.corpus.idx2word], f)


if __name__ == '__main__':
    file_name = download_data(URL, EXPECTED_BYTES, LOCAL_DATA_DIR)
    word_list = get_word_list(file_name)
    corpus = Corpus(word_list)
    cbownet = Cbownet(corpus)
    cbownet.train()