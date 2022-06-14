from __future__ import unicode_literals, print_function, division
import numpy as np
from collections import Counter


class GSDPMM:
    def __init__(self, dataset, val_k, beta):
        self.K = val_k
        self.alpha = 1.0
        self.beta = beta
        self.iterNum = 30
        self.docu_set = docu_set(dataset)
        self.number_docs = len(dataset)
        self.V = self.docu_set.V
        self.alpha0 = self.K * self.alpha
        self.beta0 = self.V * self.beta
        self.m_z = np.zeros(self.K, dtype=np.int)
        self.n_z = np.zeros(self.K, dtype=np.int)
        self.n_zv = np.zeros([self.K, self.V], dtype=np.int)
        self.z_c = np.zeros(self.number_docs, dtype=np.int)
        self.num_list = self.docu_set.num_list
        self.wordid_array = self.docu_set.wordid_array
        self.wordfreq_array = self.docu_set.wordfreq_array
        self.largedouble = 1e100
        self.smalldouble = 1e-100

    def initialize(self):
        for d in range(self.number_docs):
            self.z_c[d] = int(np.floor(self.K * np.random.uniform()))
            cluster = self.z_c[d]
            self.m_z[cluster] = self.m_z[cluster] + 1
            for w in range(len(self.num_list[d])):
                self.n_zv[cluster][self.num_list[d][w]] = self.n_zv[cluster][self.num_list[d][w]] + 1
                self.n_z[cluster] = self.n_z[cluster] + 1

    def gibbs_sampling(self):
        for i in range(self.iterNum):
            for d in range(self.number_docs):
                cluster = self.z_c[d]
                self.m_z[cluster] = self.m_z[cluster] - 1
                for w in range(len(self.num_list[d])):
                    self.n_zv[cluster][self.num_list[d][w]] = self.n_zv[cluster][self.num_list[d][w]] - 1
                    self.n_z[cluster] = self.n_z[cluster] - 1

                cluster = self.sample_cluster(d)
                self.z_c[d] = cluster
                self.m_z[cluster] = self.m_z[cluster] + 1
                for w in range(len(self.num_list[d])):
                    self.n_zv[cluster][self.num_list[d][w]] = self.n_zv[cluster][self.num_list[d][w]] + 1
                    self.n_z[cluster] = self.n_z[cluster] + 1
            #print('Iter ', i)

    def sample_cluster(self, d):
        prob = np.zeros(self.K)
        overflow_count = np.zeros(self.K)
        for k in range(self.K):
            prob[k] = (self.m_z[k] + self.alpha) / (self.number_docs + self.alpha0)
            value2 = 1.0
            i = 0
            for w in range(len(self.wordid_array[d])):
                wordNo = self.wordid_array[d][w]
                wordfreq = self.wordfreq_array[d][w]
                for j in range(wordfreq):
                    value2 = value2 * (self.n_zv[k][wordNo] + self.beta + j) / (self.n_z[k] + self.beta0 + i)
                    i = i + 1
                    if value2 < self.smalldouble:
                        overflow_count[k] = overflow_count[k] - 1
                        value2 = value2 * self.largedouble
            prob[k] = prob[k] * value2
        self.recompute_prob(prob, overflow_count, self.K)
        for k in range(1, self.K):
            prob[k] = prob[k - 1] + prob[k]

        sample = np.random.uniform() * prob[self.K - 1]
        kchoosed = 0
        for kchoosed in range(self.K):
            if sample < prob[kchoosed]:
                break
        return kchoosed

    def recompute_prob(self, prob, overflow_count, K):
        max_common = -1e20
        for k in range(K):
            if overflow_count[k] > max_common and prob[k] > 0:
                max_common = overflow_count[k]

        for k in range(K):
            if prob[k] > 0:
                prob[k] = prob[k] * pow(self.largedouble, overflow_count[k] - max_common)


class docu_set:
    def __init__(self, dataset):
        self.number_docs = 0
        self.docs = dataset
        self.wordtoId = {}
        self.wordfreq = {}
        self.V = len(self.wordtoId)
        self.num_list, self.wordid_array, self.wordfreq_array = self.convert_to_numlist()

    def convert_to_numlist(self):
        n_lines = len(self.docs)
        num_list = [[] for i in range(n_lines)]
        wordid_array = [[] for i in range(n_lines)]
        wordfreq_array = [[] for i in range(n_lines)]

        for i in range(n_lines):
            this_line = self.docs[i]
            split_line = this_line.split()
            for j in range(len(split_line)):
                if split_line[j] in self.wordtoId:
                    self.wordfreq[self.wordtoId[split_line[j]]] = self.wordfreq[self.wordtoId[split_line[j]]] + 1
                    Id = self.wordtoId.get(split_line[j])
                    if Id in wordid_array[i]:
                        wordfreq_array[i][wordid_array[i].index(Id)] += 1
                    else:
                        wordid_array[i].append(Id)
                        wordfreq_array[i].append(1)

                else:
                    self.wordtoId[split_line[j]] = self.V
                    self.V = self.V + 1
                    self.wordfreq[self.wordtoId[split_line[j]]] = 1
                    Id = self.wordtoId.get(split_line[j])
                    if Id in wordid_array[i]:
                        wordfreq_array[i][wordid_array[i].index(Id)] += 1
                    else:
                        wordid_array[i].append(Id)
                        wordfreq_array[i].append(1)

                num_list[i].append(self.wordtoId[split_line[j]])
        return num_list, wordid_array, wordfreq_array

'''
def read_data_set_new(file_name):
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            a = line.split('\t')
            data.append(a[0].strip())
    return data


dataset = read_data_set_new('/Drive/git projects/CV2Datasets/resources/data/hulin_pulication_processed.txt')
gsdmm = GSDPMM(dataset)

gsdmm.initialize()

gsdmm.gibbs_sampling()
A = gsdmm.z_c
print(A)
print(type(A))
print(A.shape)
C = Counter(A)
# print(C)
cluster_index = {}

print('numbe of classes', len(C))
docu_dict = {}
for i, doc in enumerate(gsdmm.docu_set.docs):
    cls = str(A[i])
    if cls in docu_dict:
        temp = docu_dict[cls]
        temp.append(doc)
        docu_dict[cls] = temp
    else:
        docu_dict[cls] = [doc]

print(len(docu_dict))

for cls in docu_dict:
    print(cls, ' : ', docu_dict[cls])

num_list = gsdmm.num_list
m_z = gsdmm.m_z
n_z = gsdmm.n_z
n_zv = gsdmm.n_zv
docu_num = gsdmm.number_docs
z_c = gsdmm.z_c
wordid_array = gsdmm.wordid_array
wordfreq_array = gsdmm.wordfreq_array'''
