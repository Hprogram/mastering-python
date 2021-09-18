
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Util.logger import logger
import numpy as np
import TuningData
import copy
class DataSet:
    def __init__(self):
        self.features = None
        self.answer = None
        self.featuresPath = None
        self.answerPath = None
class DataLoader:
    def __init__(self, path):
        self.data = np.array([], dtype=DataSet)
        self.position = 0
        self.__load(path)
    def __isNpy(self, fileName):
        return fileName.rfind('.npy') == len(fileName) - 4    
    def __convertFileNameToAnswer(self, fileName):
        return fileName[fileName.find('_') + 1 : fileName.rfind('.')] + '.answer'
    def __isAnswerExist(self, path, fileName):
        return os.path.isfile(path+'/'+self.__convertFileNameToAnswer(fileName))
    def __readAnswerFile(self, fullPath):
        r = open(fullPath, mode='rt', encoding='utf-8')
        lines = r.readlines()
        for i in range(0, len(lines)):
            lines[i] = lines[i].strip()
        x = np.array(lines)
        y = x.astype(np.float)
        tuningData = TuningData.TuningData(y)
        tuningData.normalize()
        return tuningData.data
    def __load(self, rootPath):
        for path, folders, files in os.walk(rootPath):
            for filename in files:
                if not self.__isNpy(filename): continue
                if not self.__isAnswerExist(path, filename):
                    logger.error("Answer is not exist : " + filename)
                    continue
                fragment = DataSet()
                fragment.featuresPath = path + '/' + filename
                fragment.answerPath = path + '/' + self.__convertFileNameToAnswer(filename)
                self.data = np.append(self.data, fragment)
    def read(self, count):
        dataFrag = np.array([], dtype=DataSet)

        for i in range(self.position , self.position + count):
            if (self.position >= len(self.data)): break
            temp = copy.copy(self.data[i])
            temp.features = np.load(temp.featuresPath)
            temp.answer = self.__readAnswerFile(temp.answerPath)
            dataFrag = np.append(dataFrag, temp)
        self.position = self.position + len(dataFrag)
        return dataFrag
    def size(self):
        return len(self.data)

if (__name__ == '__main__'):
    classicLoader = DataLoader("/Users/choon/git/sample_web/python_module/FeatureExtract/LearnData/classic")

    print("test")

