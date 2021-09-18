import tensorflow as tf
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import math
import time
import shutil
from Util.logger import *
import itertools
import matplotlib.pyplot as plt
from LearningModule import TuningData
from FeatureExtract import Normalizer
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
session = InteractiveSession(config=config)




savePng = "./Result/%s.png" %(time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time())))
modelSummary = "init model"

nb_filters = [32, 64, 128]  # number of convolutional filters to use
# pool_size = (2, 2)  # size of pooling area for max pooling
kernel_size = [(5, 5),(3,3),(3,3)]  # convolution kernel size
nb_layers = 3# 5

def getCNNModel(x_train, y_train):
    global nb_filters, kernel_size, nb_layers
    logger.debug("Get CNN Model")

    input_shape = (x_train.shape[1], x_train.shape[2], 1)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    # model.add(tf.keras.layers.BatchNormalization())

    for layer in range(nb_layers):
        model.add(tf.keras.layers.Conv2D(nb_filters[layer], kernel_size[layer], kernel_size[layer], padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        
        # model.add(tf.keras.layers.ELU(alpha=1.0))  
        # if layer == 0:
        #     model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))
        model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Activation('relu'))    
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Activation('relu'))    
    model.add(tf.keras.layers.Dense(5))

    # model.add(tf.keras.layers.ELU(alpha=1.0))

    return model


def saveModelSummryToMemory(model):
    import io
    global modelSummary
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    modelSummary = stream.getvalue()
    stream.close()
    print(modelSummary)

def getCompiledModel(x_train, y_train):
    global modelSummary
    model = getCNNModel(x_train, y_train)
    model.compile(loss="mae", optimizer='rmsprop', metrics=["mean_absolute_error"])
    
    # test = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # test = tf.keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
    # test = tf.keras.optimizers.Adagrad(learning_rate=0.001)
    # test = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
    # test = tf.keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    # model.compile(loss="mae", optimizer=test, metrics=["mean_absolute_error"])

    saveModelSummryToMemory(model)
    # model.save_weights('initmodel.h5')
    # history = model.fit(x_train, y_train, epochs=200)
    return model

def getExceptedTrainSet(wavFileDoc, checkList):
    sub_x_train = wavFileDoc['dataSet']['x_train']
    sub_y_train = wavFileDoc['dataSet']['y_train']

    deleteList = []
    for docKey in wavFileDoc:
        if wavFileDoc[docKey].get('fileName') == None : continue
        if wavFileDoc[docKey]['fileName'] not in checkList : continue
        deleteList.append(wavFileDoc[docKey]['index'])
    
    sub_x_train = np.delete(sub_x_train, deleteList, 0)
    sub_y_train = np.delete(sub_y_train, deleteList, 0)
    sub_x_train = sub_x_train.reshape(sub_x_train.shape[0], sub_x_train.shape[1], sub_x_train.shape[2], 1)
    return sub_x_train, sub_y_train

def getTestSet(wavFileDoc, checkList):
    x_train = wavFileDoc['dataSet']['x_train']
    y_train = wavFileDoc['dataSet']['y_train']

    ret_x_train = []
    ret_y_train = []
    for docKey in wavFileDoc:
        if wavFileDoc[docKey].get('fileName') == None : continue
        if wavFileDoc[docKey]['fileName'] not in checkList : continue
        if wavFileDoc[docKey]['increaseIndex'] == 0:
            x_test = x_train[wavFileDoc[docKey]['index']]
            x_test = x_test.reshape(1, x_test.shape[0], x_test.shape[1], 1)
            y_test = y_train[wavFileDoc[docKey]['index']]
            y_test = y_test.reshape(1, y_test.shape[0])
            ret_x_train.append(x_test.copy())
            ret_y_train.append(y_test.copy())
    if len(ret_x_train) == 0:
        return None, None
    return ret_x_train, ret_y_train
def sampleTest(y_train):
    
    count_set = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for item in y_train:
        count_set[int(round(item[4]))] = count_set[int(round(item[4]))] + 0.2
    for i, count in enumerate(count_set):
        print(str(i) + "  " + str(count))

    

def getAugmentationSize(wavFileDoc):
    augmentSize = 0
    for docKey in wavFileDoc:
        if wavFileDoc[docKey].get('increaseIndex') == None : continue
        if augmentSize < wavFileDoc[docKey]['increaseIndex']: 
            augmentSize = wavFileDoc[docKey]['increaseIndex']
    return augmentSize + 1

def getOriginXTrainCount(wavFileDoc):
    countMap = {}
    for docKey in wavFileDoc:
        if wavFileDoc[docKey].get('increaseIndex') == None : continue
        if wavFileDoc[docKey].get('fileName') == None : continue
        countMap[wavFileDoc[docKey].get('fileName')] = 0
    return len(countMap)

def drawGraphTest(y_train):
    subjects = ["Threshold", "Ratio", "Attack", "Release", "Gain"]
    plt.figure(figsize=(20,20), dpi=100)
    normalizer = Normalizer.Normalizer()
    # for j, y_train_item in enumerate(y_train):
    #     y_train[j] = normalizer.standardization(y_train_item)
    plt.scatter(y_train.T[0], y_train.T[4])
    plt.show()
    plt.savefig("test1.png")


def kCrossoverLearing():
    wavFileDoc = np.load("./data.npy", allow_pickle=True).item()
    drawGraphTest(wavFileDoc['dataSet']['y_train'])
    x_train = wavFileDoc['dataSet']['x_train']
    y_train = wavFileDoc['dataSet']['y_train']
    sampleTest(y_train)
    totalSize = getOriginXTrainCount(wavFileDoc)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    learningResult = np.zeros((totalSize, 2, 5), dtype=np.float64)
    model = getCompiledModel(x_train, y_train)
    init_weights = model.get_weights()
    
    checkedFileName = {}
    i = 0
    normalizer = Normalizer.Normalizer()
    checkList = []
    # import random
    # random.seed(119)
    # keys =  list(wavFileDoc.keys())  
    # random.shuffle(keys)
    # for docKey in keys:
    for docKey in wavFileDoc:
        if wavFileDoc[docKey].get('fileName') == None : continue
        if checkedFileName.get(wavFileDoc[docKey]['fileName']) != None : continue
        checkList.append(wavFileDoc[docKey]['fileName'])
        checkedFileName[wavFileDoc[docKey]['fileName']] = True
        if len(checkList) < 30 and len(checkedFileName) != totalSize: continue

        sub_x_train, sub_y_train = getExceptedTrainSet(wavFileDoc, checkList)

        for j, y_train_item in enumerate(sub_y_train):
            sub_y_train[j] = normalizer.standardization(y_train_item)

        x_tests, y_tests = getTestSet(wavFileDoc, checkList)
        for y_test in y_tests:
            temp = normalizer.standardization(y_test.reshape(y_test.shape[1]))
            y_test = temp.reshape(1, temp.shape[0])

        model.set_weights(init_weights)
        # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        # history = model.fit(sub_x_train, sub_y_train, epochs=100, callbacks=[callback], validation_split=0.2)
        # history = model.fit(sub_x_train, sub_y_train, epochs=100, validation_split=0.2)
        history = model.fit(sub_x_train, sub_y_train, epochs=33)
        # model.save("test_model.h5")

        
        answers = model.predict(np.array(x_tests).reshape(len(x_tests),x_tests[0].shape[1],x_tests[0].shape[2],x_tests[0].shape[3]))
        for index in range(0,len(x_tests)):
            learningResult[i+index][0] = normalizer.unStandardization(y_tests[index])
            learningResult[i+index][1] = normalizer.unStandardization(answers[index].reshape(1,answers[index].shape[0]))

        i = i + len(checkList)
        logger.debug(str(checkList) + '               index : ' + str(i))
        checkList = []
        tf.keras.backend.clear_session()
        
    # np.save("./resultData.npy",learningResult)
    if drawGraph(learningResult):
        model.save(savePng.replace('.png', '.h5'))
        model.set_weights(init_weights)
        model.save(savePng.replace('.png', '_init.h5'))

def _getTotalLoss(a, b):
    totalLoss = 0
    totalScore = 0

    for i in range (0, len(a)):
        totalLoss = totalLoss + abs(a[i] - b[i])
        totalScore += abs(a[i])

    return "%.3f - %.3f %%" % (totalLoss, (totalScore - totalLoss) / totalScore * 100)


def drawGraph(data):
    global savePng
    subjects = ["Threshold", "Ratio", "Attack", "Release", "Gain"]
    plt.figure(figsize=(20,20), dpi=100)
    percents =[]
    for i, subject in enumerate(subjects):

        plt.subplot(5,1,i+1)
        #keras.losses.mean_absolute_percentage_error(y_true, y_pred)
        aa = tf.keras.losses.mean_absolute_percentage_error(data[:,0,i], data[:,1,i])
        percents.append(100-aa.numpy())
        plt.title(subject + "_" + str(_getTotalLoss(data[:,0,i], data[:,1,i])) + "_" + str(100-aa.numpy()) + "%")
        plt.plot(data[:,0,i],'r', label='real')
        plt.plot(data[:,1,i],'b', label='predict')
        plt.legend(loc="best")
    
    # global modelSummary
    # plt.subplot(6,1,6)
    # plt.title("model structure")
    # plt.text(0, 0, modelSummary, wrap=True)
    
    savePng = savePng.replace('./Result/', './Result/'+str(round(percents[0]+percents[4],1))+"_")
    plt.show()
    plt.savefig(savePng)
    plt.close('all')

    return round(percents[0]+percents[4],1) >= 125


def copyModelWrapper():
    global savePng
    shutil.copy2('LearningModule/ModelWrapper.py', savePng.replace(".png", ".py"))


if (__name__ == '__main__'):
    # global nb_filters, kernel_size, nb_layers, savePng\
    available_nb_filters = [32, 64, 128, 256]
    available_kernel_size = [3,5,7]

    from itertools import combinations_with_replacement
    from itertools import product

    itemCnt = 0
    for next_nb_layers in range(4,7):
        nb_layers = next_nb_layers
        permu_kernel_size = []
        permu_nb_filters = []

        for cwr in combinations_with_replacement(available_kernel_size, next_nb_layers):
            permu_kernel_size.append(cwr)
        
        for cwr in combinations_with_replacement(available_nb_filters, next_nb_layers):
            permu_nb_filters.append(cwr)
        
        
        for item in product(permu_kernel_size, permu_nb_filters):
            itemCnt += 1
            if itemCnt <= 2352: continue
            kernel_size = item[0]
            nb_filters = item[1]
            nb_layers = len(nb_filters)
            savePng = "./Result/%s_%s_%s_%s.png" %(time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time())), str(nb_filters), str(kernel_size), itemCnt)
            # copyModelWrapper()
            print (savePng)
            kCrossoverLearing()
            # resultData = np.load("./resultData.npy", allow_pickle=True)
            print ("TEST   :  "+ str(item)) 
            print (savePng) 
            # drawGraph(resultData)
            # break
    print ("Done")
            
