import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class arrhythmia_dataset_dev(Dataset):
    def __init__(self,train=True):
        if train :
            self.data = np.array(pd.read_csv('./数据集/diff_train.csv').iloc[:,np.arange(56)])
            label_not_tran = pd.read_csv('./数据集/diff_train.csv')['label']
            self.label = np.array(label_not_tran.map({0: 0, 1: 1}))
        else:
            self.data = np.array(pd.read_csv('./数据集/diff_test.csv').iloc[:, np.arange(56)])
            label_not_tran = pd.read_csv('./数据集/diff_test.csv')['label']
            self.label = np.array(label_not_tran.map({0: 0, 1: 1}))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)


class arrhythmia_dataset_equal(Dataset):
    def __init__(self,train=True):
        if train :
            self.data = np.array(pd.read_csv('./数据集/mdiff_train.csv').iloc[:, np.arange(56)])
            label_not_tran = pd.read_csv('./数据集/mdiff_train.csv')['label']
            self.label = np.array(label_not_tran.map({0: 0, 1: 1}))
        else:
            self.data = np.array(pd.read_csv('./数据集/mdiff_test.csv').iloc[:, np.arange(56)])
            label_not_tran = pd.read_csv('./equa数据集l/mdiff_test.csv')['label']
            self.label = np.array(label_not_tran.map({0: 0, 1: 1}))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

class arrhythmia_dataset_dev_val(Dataset):
    def __init__(self):
        self.data = np.array(pd.read_csv('./数据集/diff_dev.csv').iloc[:, np.arange(56)])
        label_not_tran = pd.read_csv('./数据集/diff_dev.csv')['label']
        self.label = np.array(label_not_tran.map({0: 0, 1: 1}))


class arrhythmia_dataset_equal_val(Dataset):
    def __init__(self):
        self.data = np.array(pd.read_csv('./数据集/mdiff_dev.csv').iloc[:, np.arange(56)])
        label_not_tran = pd.read_csv('./数据集/mdiff_dev.csv')['label']
        self.label = np.array(label_not_tran.map({0: 0, 1: 1}))
        # else:
        #     self.data = np.array(pd.read_csv('./equal/test.csv').iloc[:, np.arange(56)])
        #     label_not_tran = pd.read_csv('./equal/test.csv')['label']
        #     self.label = np.array(label_not_tran.map({0: 0, 1: 1}))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

