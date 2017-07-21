# coding=utf-8
import numpy as np
from config import config
import csv


class Dataset(object):
    def __init__(self, data_source):
        self.data_source = data_source
        self.index_in_epoch = 0
        self.alphabet = config.alphabet
        self.alphabet_size = config.alphabet_size
        self.num_classes = config.nums_classes
        self.l0 = config.l0
        self.epochs_completed = 0
        self.batch_size = config.batch_size
        self.example_nums = config.example_nums
        self.doc_image = []
        self.label_image = []

    def next_batch(self):
        # 得到Dataset对象的batch
        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size
        if self.index_in_epoch > self.example_nums:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.example_nums)
            np.random.shuffle(perm)
            self.doc_image = self.doc_image[perm]
            self.label_image = self.label_image[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = self.batch_size
            assert self.batch_size <= self.example_nums
        end = self.index_in_epoch
        batch_x = np.array(self.doc_image[start:end], dtype='int64')
        batch_y = np.array(self.label_image[start:end], dtype='float32')

        return batch_x, batch_y

    def dataset_read(self):
        # doc_vec表示一个一篇文章中的所有字母，doc_image代表所有文章
        # label_class代表分类
        # doc_count代表数据总共有多少行
        docs = []
        label = []
        doc_count = 0
        csvfile = open(self.data_source, 'r')
        for line in csv.reader(csvfile, delimiter=',', quotechar='"'):
            content = line[1] + ". " + line[2]
            docs.append(content.lower())
            label.append(line[0])
            doc_count = doc_count + 1

        # 引入embedding矩阵和字典
        print "引入嵌入词典和矩阵"
        embedding_w, embedding_dic = self.onehot_dic_build()

        # 将每个句子中的每个字母，转化为embedding矩阵的索引
        # 如：doc_vec表示一个一篇文章中的所有字母，doc_image代表所有文章
        doc_image = []
        label_image = []
        print "开始进行文档处理"
        for i in range(doc_count):
            doc_vec = self.doc_process(docs[i], embedding_dic)
            doc_image.append(doc_vec)
            label_class = np.zeros(self.num_classes, dtype='float32')
            label_class[int(label[i]) - 1] = 1
            label_image.append(label_class)

        del embedding_w, embedding_dic
        print "求得训练集与测试集的tensor并赋值"
        self.doc_image = np.asarray(doc_image, dtype='int64')
        self.label_image = np.array(label_image, dtype='float32')

    def doc_process(self, doc, embedding_dic):
        # 如果在embedding_dic中存在该词，那么就将该词的索引加入到doc的向量表示doc_vec中，不存在则用UNK代替
        # 不到l0的文章，进行填充，填UNK的value值，即0
        min_len = min(self.l0, len(doc))
        doc_vec = np.zeros(self.l0, dtype='int64')
        for j in range(min_len):
            if doc[j] in embedding_dic:
                doc_vec[j] = embedding_dic[doc[j]]
            else:
                doc_vec[j] = embedding_dic['UNK']
        return doc_vec

    def onehot_dic_build(self):
        # onehot编码
        alphabet = self.alphabet
        embedding_dic = {}
        embedding_w = []
        # 对于字母表中不存在的或者空的字符用全0向量代替
        embedding_dic["UNK"] = 0
        embedding_w.append(np.zeros(len(alphabet), dtype='float32'))

        for i, alpha in enumerate(alphabet):
            onehot = np.zeros(len(alphabet), dtype='float32')
            embedding_dic[alpha] = i + 1
            onehot[i] = 1
            embedding_w.append(onehot)

        embedding_w = np.array(embedding_w, dtype='float32')
        return embedding_w, embedding_dic

# 如果运行该文件，执行此命令，否则略过
if __name__ == "__main__":
    data = Dataset("data/ag_news_csv/train.csv")
    data.dataset_read()