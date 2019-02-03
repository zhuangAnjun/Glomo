import numpy as np
import keras
import os
from collections import Counter
from configs.config import c as config

home = os.getcwd()
data_dir = os.path.join(home, "datasets", "wiki", "wiki2_train")
vocab_dir = os.path.join(home, "data", "vocb")

# build vocabulary
def build_vocab(data_dir, vocab_dir, vocab_size=config.VOCAB_SIZE):
    """根据训练集构建词汇表，存储"""
    data_train = open(data_dir,'r',encoding='utf8').readlines();
    all_data = []
    for content in data_train:
        all_data.extend(content.split())

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))

    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open(vocab_dir, 'w', encoding='utf-8', errors='ignore').write('\n'.join(words) + '\n')
    print('build vocabulary complete')


def read_vocab(vocab_dir):
    """读取词汇表"""
    with open(vocab_dir,'r',encoding='utf8') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def process_article(article, word_to_id, max_length=config.SENTENCE_LENGTH):
    """将文件转换为id表示"""

    data_id = []
    for i in range(len(article)):
        data_id.append([int(word_to_id[x]) for x in article[i] if x in word_to_id])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = keras.preprocessing.sequence.pad_sequences(data_id, max_length)

    return x_pad


def generator_train(data_train, vocab_dir, batch_size=config.BATCH_SIZE):
    """ 生成器 """
    num=0
    article = []
    flag = ['=', '\n', ' ']

    word, word_to_id = read_vocab(vocab_dir)

    while True:
        with open(data_train,'r',encoding='utf8') as file:
            lines = file.readlines()
            for line in lines:
                if len(line)<=1:
                    continue
                if line[1] in flag:
                    continue
                line = line.split()
                article.append(line)
                num = num+1
                if num%batch_size == 0:

                    article = process_article(article, word_to_id)
                    yield np.array(article)

                    article = []

def generator_valid(data_val, vocab_dir, batch_size=config.BATCH_SIZE):
    """ 生成器 """
    num=0
    article = []
    flag = ['=', '\n', ' ']

    word, word_to_id = read_vocab(vocab_dir)

    while True:
        with open(data_val,'r') as file:
            lines = file.readlines()
            for line in lines:
                if line[1] in flag:
                    continue;
                line = line.split()
                article.append(line)
                num = num+1
                if num%batch_size == 0:

                    article = process_article(article, word_to_id)
                    yield np.array(article)

                    article = []

# 先建立词表
# build_vocab(data_dir, vocab_dir)