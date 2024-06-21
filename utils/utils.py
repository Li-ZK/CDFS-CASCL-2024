import numpy as np
import random
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import logging
import datetime

import torch
import torch.nn as nn

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def weights_init(m):
    if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

import torch.utils.data as data

class matcifar(data.Dataset):
    def __init__(self, imdb, train, d, medicinal):

        self.train = train  # training set or test set
        self.imdb = imdb
        self.d = d
        self.x1 = np.argwhere(self.imdb['set'] == 1)
        self.x2 = np.argwhere(self.imdb['set'] == 3)
        self.x1 = self.x1.flatten()
        self.x2 = self.x2.flatten()

        if medicinal == 1:
            self.train_data = self.imdb['data'][self.x1, :, :, :]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][self.x2, :, :, :]
            self.test_labels = self.imdb['Labels'][self.x2]

        else:
            self.train_data = self.imdb['data'][:, :, :, self.x1]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][:, :, :, self.x2]
            self.test_labels = self.imdb['Labels'][self.x2]
            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))
                self.test_data = self.test_data.transpose((3, 2, 0, 1))
            else:
                self.train_data = self.train_data.transpose((3, 0, 2, 1))
                self.test_data = self.test_data.transpose((3, 0, 2, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:

            img, target = self.train_data[index], self.train_labels[index]
        else:

            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def sanity_check(all_set):
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        if len(all_set[class_]) >= 200:
            all_good[class_] = all_set[class_][len(all_set[class_])-200:]
            nclass += 1
            nsamples += len(all_good[class_])
    print('the number of class:', nclass)
    print('the number of sample:', nsamples)
    return all_good

def flip(data):
    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data

def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key]
    GroundTruth = label_data[label_key]

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))
    data_scaler = preprocessing.scale(data.astype(float))  # (X-X_mean)/X_std
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth

def load_data_houston(image_file, label_file,label_file1):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    label_data1 = sio.loadmat(label_file1)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    label_key1 = label_file1.split('/')[-1].split('.')[0]

    data_all = image_data[data_key]  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth_train = label_data[label_key]
    GroundTruth_test = label_data1[label_key1]


    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  #标准化 (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth_train, GroundTruth_test  # image:(512,217,3),label:(512,217)

def classification_map(map, groundTruth, dpi, savePath):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)

    return 0

def preprocess(num_ways, num_shots, num_queries, batch_size, device):
    """
    prepare for train and evaluation
    :param num_ways: number of classes for each few-shot task
    :param num_shots: number of samples for each class in few-shot task
    :param num_queries: number of queries for each class in few-shot task
    :param batch_size: how many tasks per batch
    :param device: the gpu device that holds all data
    :return: number of samples in support set
             number of total samples (support and query set)
             mask for edges connect query nodes
             mask for unlabeled data (for semi-supervised setting)
    """
    # set size of support set, query set and total number of data in single task
    num_supports = num_ways * num_shots # 9 * 1 = 9
    num_samples = num_supports + num_queries * num_ways #  9 * 1 + 19 * 9 = 180

    # set edge mask (to distinguish support and query edges) 设置边掩码（用于区分支持和查询边）
    support_edge_mask = torch.zeros(batch_size, num_samples, num_samples).to(device)
    support_edge_mask[:, :num_supports, :num_supports] = 1
    query_edge_mask = 1 - support_edge_mask
    evaluation_mask = torch.ones(batch_size, num_samples, num_samples).to(device) # 作用？mask for unlabeled data (for semi-supervised setting)
    return num_supports, num_samples, query_edge_mask, evaluation_mask

def set_logging_config(logdir, num_seeds):
    myTimeFormat = '%Y-%m-%d_%H-%M-%S'
    nowTime = datetime.datetime.now().strftime(myTimeFormat)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logging.basicConfig(format="[%(asctime)s] [%(levelname)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, str(num_seeds) +'seeds_'+nowTime+'.log')),
                                  logging.StreamHandler(os.sys.stdout)])

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits
