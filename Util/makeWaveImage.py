import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Util.logger import *  
import librosa
import librosa.display
import matplotlib.pyplot as plt

def makeWaveImageAll(rootPath):
    plt.rcParams['agg.path.chunksize'] = 100000
    for path, folders, files in os.walk(rootPath):
        for filename in files:
            if '.wav' not in filename: continue
            if 'classic_6' in filename: continue
            if 'classic_20' in filename: continue
            if 'tuned' not in filename: continue
            y, sr = librosa.load(path + '/' + filename, sr=22050, mono=False)
            
            plt.figure(figsize=(20,4), dpi=100)
            # plt.figure()
            # plt.subplot(3,1,1)
            librosa.display.waveplot(y, sr=sr)
            plt.title('Stereo')
            pngFileName = filename.replace('.wav', '.png')
            logger.debug(path + '/' + pngFileName)
            plt.savefig(path + '/' + pngFileName)
            plt.close('all')
    return ;

if (__name__ == '__main__'):
    logger.debug("Start Macro")
    makeWaveImageAll('./Test/check/')
    # makeWaveImageAll('/home/deepmaster/deepmaster_audio/after_all/')
    logger.debug("Complete")
