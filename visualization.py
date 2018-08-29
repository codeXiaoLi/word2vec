# -*- coding: utf-8 -*-
import pickle
from sklearn.manifold import TSNE
from matplotlib import pylab as plt
from const import *

def plot(embeddings, labels, save_path):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(16, 9))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i,:]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(save_path)
    plt.show()
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--method', '-m', type=str, help='cbow or skipgram', default='skipgram')   # cbow or skipgram
    args = parser.parse_args()
    method = args.method
    filename = RESULT_DIR + 'final_embeddings_%s.pkl' % method
    with open(filename, 'rb') as f:
        [final_embeddings, idx2word] = pickle.load(f)
    plot_only = 50
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(final_embeddings[:plot_only, :])
    num_points = len(two_d_embeddings)
    print(num_points)
    words = [idx2word[i] for i in range(num_points)]
    plot(two_d_embeddings, words, save_path=RESULT_DIR+'embeddings_%s.png' % method)
