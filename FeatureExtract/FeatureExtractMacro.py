import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import librosa
import numpy as np
from FeatureGetter import *
from Util.logger import *
from LearningModule import TuningData
import openpyxl

class AnswerFinder:
    def __init__(self, filePath):
        wb = openpyxl.load_workbook(filePath)
        self.data = {}
        for sheetName in wb.get_sheet_names():
            ws = wb.get_sheet_by_name(sheetName)
            rowNoneCounter = 0
            for r in ws.rows:
                fileName = r[0].value
                if type(fileName) == type(None): 
                    rowNoneCounter+=1
                    if rowNoneCounter > 3: break
                    continue
                rowNoneCounter = 0
                self.data[fileName] = { 'threshold':r[3].value,
                                        'ratio':r[4].value,
                                        'attack':r[5].value,
                                        'release':r[6].value,
                                        'gain':r[7].value}
        # self.bunpo
        self.std = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for index, fileName in enumerate(self.data):
            if type(fileName) != type(""): continue
            if '.wav' not in fileName: continue
            self.std[round(self.data[fileName]['gain'])] +=1
        
    def getAnswer(self, fileName):
        answer = self.data.get(fileName)
        if answer == None: answer = {}
        return answer
    def getAguSize(self, fileName):
        if round(self.data[fileName]['gain']) == 19:
            print("test")
        size =  round(max(self.std)/self.std[round(self.data[fileName]['gain'])]) * 2
        if size < 1: return 2
        return min(size, 15)
def loadWavFiles(rootPath, answerFinder):
    wavFileDoc = {}
    for path, folders, files in os.walk(rootPath):
        for filename in files:
            if '.wav' not in filename: continue
            
            # for i in range(0,answerFinder.getAguSize(filename)):
            for i in range(0,5):
                docKey = filename + '#' + str(i)
                wavFileDoc[docKey] = {}
                wavFileDoc[docKey]['path'] = path+'/'+filename
                wavFileDoc[docKey]['fileName'] = filename
                wavFileDoc[docKey]['increaseIndex'] = i
    return wavFileDoc

def getWavFeature(filePath, increaseIndex = 0):
    y, sr = librosa.load(filePath, sr=22050, mono=True)
    featureGetter = FeatureCollecter()
    return featureGetter.getFeatures(y,sr, increaseIndex)

def makeWavFileDoc():
    # answerFinder = AnswerFinder('./FeatureExtract/LearnData_22k_mono/200415DEEP MASTER_v3.xlsx')
    # wavFileDoc = loadWavFiles('./FeatureExtract/LearnData_22k_mono', answerFinder)
    answerFinder = AnswerFinder('./music/2001415_v2.xlsx')
    wavFileDoc = loadWavFiles('./music', answerFinder)
    
    deleteList = []
    for docKey in wavFileDoc:
        wavFileDoc[docKey]['answer'] = answerFinder.getAnswer(wavFileDoc[docKey]['fileName'])
        if wavFileDoc[docKey]['answer'] == {} : 
            deleteList.append(docKey)
            continue
        print(wavFileDoc[docKey]['path'])
        wavFileDoc[docKey]['feature'] = getWavFeature(wavFileDoc[docKey]['path'], wavFileDoc[docKey]['increaseIndex'])

    for docKey in deleteList:
        del wavFileDoc[docKey]     

    featureSize = len(wavFileDoc)
    wavFileDoc["dataSet"] = {}
    wavFileDoc["dataSet"]["x_train"] = np.zeros((featureSize, 431, 128), dtype=np.float64)
    wavFileDoc["dataSet"]["y_train"] = np.zeros((featureSize, 5), dtype=np.float64)
    for i, docKey in enumerate(wavFileDoc):
        if wavFileDoc[docKey].get('answer') == None : continue
        if wavFileDoc[docKey]['answer'] == {} : continue
        wavFileDoc[docKey]['index'] = i
        wavFileDoc["dataSet"] ["x_train"][i,::] = wavFileDoc[docKey]['feature']
        data = np.array([
            wavFileDoc[docKey]['answer']['threshold'],
            wavFileDoc[docKey]['answer']['ratio'],
            wavFileDoc[docKey]['answer']['attack'],
            wavFileDoc[docKey]['answer']['release'],
            wavFileDoc[docKey]['answer']['gain']])
        tuningData = TuningData.TuningData(data)
        # tuningData.gain = tuningData.gain - 2
        # if tuningData.gain < 0 : tuningData.gain = 0
        # tuningData.normalize()
        wavFileDoc["dataSet"] ["y_train"][i,:] = tuningData.data

    return wavFileDoc


if (__name__ == '__main__'):
    doc = makeWavFileDoc()
    np.save("./data.npy",doc)
    wavFileDoc =np.load("./data.npy", allow_pickle=True).item()