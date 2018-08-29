# word2vec
Tensorflow implementation of word2vec of paper [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf), 
Including 2 models: CBOW and Skip-Gram, both of which are using negative sampling to accelerate.

## Requirements
* Python 3.6
* TensorFlow 1.8
* numpy
* ...

## Explination of dirs and files
* dirs
   * data: store train data and evaluate data
   * result: store well-trained word vector and visualization images
* files
   * const.py: store global constant
   * dataset.py: download data, read file and build corpus
   * 1_CBOW.py: build and train CBOW model
   * 2_Skipgram.py: build and train Skip-Gram model
   * visualization.py: visualization of word vectors of CBOW or Skip-Gram model

## data
* For train:  
I use text8(http://mattmahoney.net/dc/text8.zip). Its length is 17005207 and its  vacab size is 253854.
* For evaluation:  
Based on question-word.txt(https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip), which contains 8869 semantic cases and 10676 syntactic cases.  
I delete the cases whose word is not in my vocab and then get 506 semantic cases and 8946 syntactic cases to evaluation.

## Train
> How to train
> * train CBOW model: python 1_CBOW.py
> * train Skip-Gram model: python 2_Skipgram.py
* Fine-tune params:  
  By now, the best accuracy I have got is about 10%, with params below:
   * Dim of word vector: 30
   * num of negative samples: 100
   * Batchsize: 256
   * Winsize: 4(before:2, after:2)
  
  I found some regularitiers of the params(to be written down), they are quiet different from the paper. The reason may be the paper use big corpus, filter low-freq words and other details(need to do more experiment).
* Comparison between CBOW and Skip-Gram
   * CBOW is faster than Skip-Gram

## Evaluate
1. Use the analogical reasoning tasks  
By now, the best accuracy I have got is about 10% (to be improved).  
2. Find the topk similar words of the centerword:  
eg:
> Nearest to two: three, five, one, nine, six

## Visualization
* visualize of word vectors of CBOW model: python visualization.py -m=cbow
* visualize of word vectors of Skip-Gram model: python visualization.py -m=skipgram
![image](https://github.com/CSXiaoLi/word2vec/blob/master/result/embeddings_cbow.png)
