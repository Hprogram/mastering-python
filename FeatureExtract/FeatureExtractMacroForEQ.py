import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import librosa
import numpy as np
from FeatureGetter import *
from Util.logger import *
from LearningModule import TuningData
import openpyxl

def parseHighPass(str):
    if str == None: return 0
    # str = '30HZ HIGHPASS FILTER'
    pos = str.find('HZ')
    if pos == -1: return 0
    return int(str[0:pos])
    
def parseHighShelf(str):
    if str == None: return 0, 0
    # str = '6.5KHZ HIGH SHELF +1.5DB'
    hzPos = str.find('KHZ')
    if hzPos == -1: return 0, 0
    shelf = str.find('SHELF')
    if shelf == -1: return 0, 0
    return float(str[0:hzPos]) * 1000, float(str[shelf + 6:-2])
def hasK(str):
    if (str.find('k') == -1) & (str.find('K') == -1):
         return False
    return True
def parseBellType(str):
    if str == None: return [[0.0,0.0],[0.0,0.0],[0.0,0.0]]
    if '200 + 0.7/800 -0.8 /2k +0.6' == str:
        str = '200 +0.7/800 -0.8 /2k +0.6'
    if '120  +.,7 / 500 -1.0  /1.0k  +0.7' == str:
        str = '120 +0.7 / 500 -1.0 /1.0k +0.7'
    if '700k' in str:
        str = str.replace('700k', '700')
    if '3000' in str:
        str = str.replace('3000', '300')

    str= str.replace("+"," +")
    str= str.replace("-"," -")
    str= str.replace("   "," ")
    str= str.replace("  "," ")

    if str == '180 +1.0db/900 +1.0db 2.5k -1.0db':
        str = '180 +1.0db/900 +1.0db/2.5k -1.0db'

    # str = '120 +1.5db/500 -1db /1.2k +0.5db'
    bells = str.split('/')
    for i in range(3):
        try:
            bells[i] = bells[i].strip(' db')
            bells[i] = bells[i].split(' ')
            k = hasK(bells[i][0])
            if k == True: bells[i][0] = bells[i][0].strip('kK')
            bells[i][0] = float(bells[i][0])
            if k == True: bells[i][0] = bells[i][0] * 1000
            bells[i][1] = float(bells[i][1])
        except IndexError:
            bells.append([0.0,0.0])
    return bells

class AnswerFinder:
    def __init__(self, filePath):
        wb = openpyxl.load_workbook(filePath)
        self.data = {}
        for sheetName in wb.get_sheet_names():
            if 'Data Distribution' == sheetName: continue
            ws = wb.get_sheet_by_name(sheetName)
            rowNoneCounter = 0
            for r in ws.rows:
                fileName = r[0].value
                if type(fileName) == type(None): 
                    rowNoneCounter+=1
                    if rowNoneCounter > 3: break
                    continue
                rowNoneCounter = 0
                highpass = parseHighPass(r[8].value)
                highshelf = parseHighShelf(r[12].value)
                belltypes = parseBellType(r[16].value) 
                self.data[fileName] = {
                    'highpass':highpass,
                    'highshelf':highshelf[0],
                    'highshelfDB':highshelf[1],
                    'belltype1':belltypes[0][0],
                    'belltype1DB':belltypes[0][1],
                    'belltype2':belltypes[1][0],
                    'belltype2DB':belltypes[1][1],
                    'belltype3':belltypes[2][0],
                    'belltype3DB':belltypes[2][1],
                }
                # self.data[fileName] = { 'threshold':r[3].value,
                #                         'ratio':r[4].value,
                #                         'attack':r[5].value,
                #                         'release':r[6].value,
                #                         'gain':r[7].value}
        
    def getAnswer(self, fileName):
        answer = self.data.get(fileName)
        if answer == None: answer = {}
        return answer

def loadWavFiles(rootPath):
    wavFileDoc = {}
    for path, folders, files in os.walk(rootPath):
        for filename in files:
            if '.wav' not in filename: continue
            for i in range(0,5):
                docKey = filename + '#' + str(i)
                wavFileDoc[docKey] = {}
                wavFileDoc[docKey]['path'] = path+'/'+filename
                wavFileDoc[docKey]['fileName'] = filename
                wavFileDoc[docKey]['increaseIndex'] = i
    return wavFileDoc

# def getWavFeature(filePath, increaseIndex = 0):
#     y, sr = librosa.load(filePath, sr=22050, mono=True)
#     featureGetter = FeatureCollecter()
#     return featureGetter.getEQFeature(y,sr)
def getWavFeature(filePath, increaseIndex = 0):
    y, sr = librosa.load(filePath, sr=22050, mono=True)
    featureGetter = FeatureCollecter()
    return featureGetter.getFeatures(y,sr, increaseIndex)

def makeWavFileDoc():
    # answerFinder = AnswerFinder('./FeatureExtract/LearnData_22k_mono/200415DEEP MASTER_v3.xlsx')
    # wavFileDoc = loadWavFiles('./FeatureExtract/LearnData_22k_mono')
    answerFinder = AnswerFinder('./music/2001415_v2.xlsx')
    wavFileDoc = loadWavFiles('./music')

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
    # wavFileDoc["dataSet"]["x_train"] = np.zeros((featureSize, 6000, 128), dtype=np.float64)
    wavFileDoc["dataSet"]["x_train"] = np.zeros((featureSize, 431, 128), dtype=np.float64)
    
    wavFileDoc["dataSet"]["y_train"] = np.zeros((featureSize, 9), dtype=np.float64)
    for i, docKey in enumerate(wavFileDoc):
        if wavFileDoc[docKey].get('answer') == None : continue
        if wavFileDoc[docKey]['answer'] == {} : continue
        wavFileDoc[docKey]['index'] = i
        wavFileDoc["dataSet"] ["x_train"][i,::] = wavFileDoc[docKey]['feature']
        data = np.array([
            wavFileDoc[docKey]['answer']['highpass'],
            wavFileDoc[docKey]['answer']['highshelf'],
            wavFileDoc[docKey]['answer']['highshelfDB'],
            wavFileDoc[docKey]['answer']['belltype1'],
            wavFileDoc[docKey]['answer']['belltype1DB'],
            wavFileDoc[docKey]['answer']['belltype2'],
            wavFileDoc[docKey]['answer']['belltype2DB'],
            wavFileDoc[docKey]['answer']['belltype3'],
            wavFileDoc[docKey]['answer']['belltype3DB']])
        wavFileDoc["dataSet"] ["y_train"][i,:] = data

    return wavFileDoc
if (__name__ == '__main__'):
    doc = makeWavFileDoc()
    np.save("./dataEQ.npy",doc)
    wavFileDoc =np.load("./dataEQ.npy", allow_pickle=True).item()