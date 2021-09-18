import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np

class EQNormalizer():
    def __init__(self):
        wavFileDoc = np.load("./dataEQ.npy", allow_pickle=True).item()
        self.y_train_T = wavFileDoc['dataSet']['y_train'].T

    def normalize(self, data):
        for i, item in enumerate(data):
            data[i] = (item - self.y_train_T[i].min()) / (self.y_train_T[i].max()- self.y_train_T[i].min())
        return data
    def unNormalize(self, data):
        for i, item in enumerate(data[0]):
            data[0][i] = item * (self.y_train_T[i].max()- self.y_train_T[i].min()) + self.y_train_T[i].min()
        return data

    def standardization(self, data):
        for i, item in enumerate(data):
            data[i] = (item - np.mean(self.y_train_T[i])) / np.std(self.y_train_T[i])
        return data
        # print('mean:', np.mean(arr))
        # print('standard deviation:', np.std(arr))
        # print('variance:', np.var(arr))

    def unStandardization(self, data):
        for i , item in enumerate(data[0]):
            data[0][i] = item * np.std(self.y_train_T[i]) + np.mean(self.y_train_T[i])
        return data