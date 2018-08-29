# -*- coding: utf-8 -*-
import os
import urllib.request
import zipfile
import numpy as np
from collections import Counter
import random
from const import *


def download_data(url, expected_bytes, local_data_dir):
    """
    :param url: data download address
    :param expected_bytes: expected size of data
    :param local_data_dir: local dir to save data
    :return: path of downloaded file
    """
    if not os.path.exists(local_data_dir):
        os.makedirs(local_data_dir)
    local_data_path = local_data_dir + os.path.split(url)[1]

    if not os.path.exists(local_data_path):
        urllib.request.urlretrieve(url, local_data_path)
    if os.stat(local_data_path).st_size == expected_bytes:
        print('Found and verified', local_data_path)
    else:
        raise Exception(local_data_path, ' is found, but the size is not right. ',
                        'found_bytes', os.stat(local_data_path).st_size, ', expected_bytes', expected_bytes,
                        'Can you get to it with a browser?')
    return local_data_path


def get_word_list(filename):
    """
    :param filename:
    :return: word_list(every word in the list is not a blank and do not include blank character)
    """
    with zipfile.ZipFile(filename) as f:
        word_list = str(f.read(f.namelist()[0]), encoding="utf8").split()
        print('Got word_list, word num:', len(word_list))
        return word_list


class Corpus(object):
    def __init__(self, word_list):
        self.word_list = word_list
        word2count = dict(Counter(self.word_list).most_common())
        self.V = len(word2count)
        self.word2idx = dict(zip(word2count.keys(), range(self.V)))
        self.idx2word = dict(zip(range(self.V), word2count.keys()))
        self.idx_list = [self.word2idx[word] for word in word_list]

    def generate_batch_skipgram(self, batch_size, window_size):
        length = len(self.idx_list)
        batch, labels = [], []
        while True:
            i = random.randint(0, length-1)
            for _ in range(window_size):
                batch.append(self.idx_list[(i + window_size//2) % length])
            for j in range(window_size + 1):
                if j != window_size // 2:
                    labels.append([self.idx_list[(i + j) % length]])
            if len(batch) == batch_size:
                yield np.array(batch), np.array(labels)
                batch, labels = [], []

    def generate_batch_cbow(self, batch_size, window_size):
        length = len(self.idx_list)
        batch, labels = [], []
        while True:
            i = random.randint(0, length-1)
            batch.append([self.idx_list[(i + j) % length] for j in range(window_size + 1) if j != window_size / 2])
            labels.append([self.idx_list[(i + window_size // 2) % length]])
            if len(batch) == batch_size:
                yield np.array(batch), np.array(labels)
                batch, labels = [], []


def preprocess_qustion_word(vocab_list):
    file_name = LOCAL_DATA_DIR + 'questions-words.txt'
    with open(file_name) as f:
        fw1 = open(LOCAL_DATA_DIR + 'semantic.txt', 'w', encoding='utf-8')
        fw2 = open(LOCAL_DATA_DIR + 'syntactic.txt', 'w', encoding='utf-8')
        fw = fw1
        for i, line in enumerate(f.readlines()):
            if line[0] == ':':
                fw = fw2 if line[2:6] == 'gram' else fw1
            else:
                words = line.strip().split()
                need_write = True
                for word in words:
                    if word not in vocab_list:
                        need_write = False
                        #print('line', i, word, 'not in vacab_list')
                        break
                    #else:
                        #print('line', i, word, 'in--------------------------')
                if need_write == True:
                    #print('write ', line)
                    fw.write(line)
    print('preprocess over')


if __name__ == '__main__':
    file_name = download_data(URL, EXPECTED_BYTES, LOCAL_DATA_DIR)
    word_list = get_word_list(file_name)
    corpus = Corpus(word_list)
    preprocess_qustion_word(list(corpus.word2idx.keys()))
    print(list(corpus.word2idx.keys())[:10])
    print(list(corpus.idx2word.keys())[:10])
    g = corpus.generate_batch_cbow(8, 4)
    print(next(g))
    print(next(g))
