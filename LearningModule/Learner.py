import DataLoader as dl
import numpy as np
import tensorflow as tf
import ModelWrapper

class TrainSetGetter:
    def __init__(self):
        self.loader = [
            dl.DataLoader("/home/deepmaster/git/sample_web/python_module/FeatureExtract/LearnData/classic"),
            dl.DataLoader("/home/deepmaster/git/sample_web/python_module/FeatureExtract/LearnData/pop"),
            dl.DataLoader("/home/deepmaster/git/sample_web/python_module/FeatureExtract/LearnData/rock"),
            dl.DataLoader("/home/deepmaster/git/sample_web/python_module/FeatureExtract/LearnData/rnb"),
            dl.DataLoader("/home/deepmaster/git/sample_web/python_module//FeatureExtract/LearnData/hiphop"),
        ]
    def __getDataSet(self, count): 
        data = np.array([], dtype=dl.DataSet)
        readCount = 0
        while readCount < count:
            isEmpty = True
            for i in range(0, len(self.loader)):
                fragment = self.loader[i].read(1)
                if len(fragment) == 0: continue
                isEmpty = False
                data = np.append(data, fragment)
                readCount = readCount + 1
                if readCount == count : break
            if isEmpty == True : break
        return data
    def __convertTrainSet(self, data):
        if len(data) == 0 : return None, None
        for i in range(0, len(data)):
            if (i == 0):
                x_train = data[0].features.reshape(1, data[0].features.shape[0] ,  data[0].features.shape[1])
                y_train = data[0].answer.reshape(1,data[0].answer.shape[0])
                continue
            x_train = np.vstack((x_train, data[i].features.reshape(1,data[0].features.shape[0] , data[0].features.shape[1])))
            y_train = np.vstack((y_train, data[i].answer.reshape(1,data[0].answer.shape[0])))
        return x_train, y_train
    def getTrainSet(self, count):
        data = self.__getDataSet(count)
        return self.__convertTrainSet(data)


if (__name__ == '__main__'):

    # np.random.shuffle(data)
    trainSetGetter = TrainSetGetter()
    x_train, y_train = trainSetGetter.getTrainSet(150)
    if type(x_train) != type(None) :    
        modelWrapper = ModelWrapper.ModelWrapper(
            activations = ['relu', 'sigmoid'], 
            optimizers = ['adam'], 
            losses = ['mse', 'mae'], 
            inputShape = [x_train.shape[1]]
        )
        modelWrapper.runCnn(x_train, y_train)
        # modelWrapper.runDnn(x_train, y_train)

    print("test")

