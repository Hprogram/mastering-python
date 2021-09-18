import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Util.logger import *  
import librosa


def sampleRateTo22k(rootPath):
    for path, folders, files in os.walk(rootPath):
        for filename in files:
            if '.wav' not in filename: continue
            y, sr = librosa.load(path + '/' + filename, sr=22050, mono=True)
            librosa.output.write_wav(path + '/' + filename, y, sr)
            logger.debug(path + '/' + filename)
    return ;

if (__name__ == '__main__'):
    logger.debug("Start Macro")
    sampleRateTo22k('./FeatureExtract/LearnData_22k_mono/')
    logger.debug("Complete")
