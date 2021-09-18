import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from TuningModule import *
from Util.logger import *  
from LearningModule import TuningData
import openpyxl

def findFilePathByIgnoreUpperCase(fileName):
    for path, folders, files in os.walk('Test/check/'):
        for target in files:
            if target.lower() == fileName.lower():
                return path, target
    return None, None

def checkSheet(ws):
    for r in ws.rows:
        fileName = r[0].value
        threshold = r[3].value
        ratio = r[4].value 
        attack = r[5].value
        release = r[6].value
        gain = r[7].value

        if type(threshold) == type(None): break
        if type(fileName) == type(None): continue
        originFilePath, originFileName = findFilePathByIgnoreUpperCase(fileName)
        print("")
        print("================file check " + fileName)
        if type(originFilePath) == type(None):
            continue
        print("file exist " + fileName)
        # Need to make tuned file

        data = [threshold,ratio,attack,release,gain]
        tuningData = TuningData.TuningData(data)
        # tuningData.gain -= 2
        f = open(originFilePath+"/"+originFileName, 'rb')
        data = f.read()
        f.close()

        y, sr = sf.read(file=io.BytesIO(data), dtype='float32')
        tuningModule2 = TuningModule(tuningData, y.T)
        masteredAudio = tuningModule2.getTunedAudio()

        f = open(originFilePath + "/" + "tuned_" + originFileName, 'wb')
        sf.write(f, masteredAudio.T, sr, format='WAV')
        f.close()

        f = open(originFilePath + "/" + originFileName.replace('wav','answer'), 'wt')
        f.writelines(str(threshold) + '\n')
        f.writelines(str(ratio) + '\n')
        f.writelines(str(attack) + '\n')
        f.writelines(str(release) + '\n')
        f.writelines(str(gain))
        f.close()

    return 

if (__name__ == '__main__'):
    # excelFileName = sys.argv[1]
    excelFileName = "Test/200415DEEP MASTER_v3.xlsx"
    print(excelFileName)
    
    # 엑셀파일 열기
    wb = openpyxl.load_workbook(excelFileName)
    for sheetName in wb.get_sheet_names():
        print(sheetName)
    for sheetName in wb.get_sheet_names():
        checkSheet(wb.get_sheet_by_name(sheetName))
