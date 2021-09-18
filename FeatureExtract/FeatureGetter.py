import time
import librosa
import numpy as np
from optparse import OptionParser
from Util.logger import *
from random import randint

class FeatureGetter():
    def __init__(self, y, sr):
        self.y = y
        self.sr = sr
    def getLoudnessTimestamp(self, sec=3):
        if (self.y.size / self.sr) < 10: return -1
        rms = self.getRMS()
        lengthOfOneSec = rms[0].size/(self.y.size / self.sr)
        windowSize = round(lengthOfOneSec * 2)

        prevSum = maxSum = sum(rms[0][0:windowSize])
        maxSumPos = 0
        for i in range(1,rms.size - windowSize +1):
            curSum = prevSum - rms[0][i-1] + rms[0][i+windowSize-1]
            prevSum = curSum
            if curSum > maxSum: 
                maxSumPos = i
                maxSum = curSum
        print(maxSumPos)
        maxSumCenterPos = round(maxSumPos + windowSize/2)
        return maxSumCenterPos / lengthOfOneSec
    def getSlicedAudio(self, left, right, timestamp):
        # rangeSec 10 
        # wantSampe = 10 * 22050
        leftWantSample = int(left * self.sr)
        rightWantSample = int(right * self.sr)

        wantSample = int(leftWantSample + rightWantSample)

        timestampPointToSample = round(timestamp*self.sr)
        if timestampPointToSample - leftWantSample < 0:
            logger.warn("Audio Slice Warnning(Left)")
            return self.y[:wantSample]
        if timestampPointToSample + rightWantSample > self.y.size:
            logger.warn("Audio Slice Warnning(Right)")
            return self.y[-wantSample:]
        return self.y[int(timestampPointToSample - leftWantSample) :int(timestampPointToSample + rightWantSample)]

    def getMel(self):
        logger.debug("")
        S = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels=32, fmax=10000)
        return S
        power_to_db = librosa.power_to_db(S, ref=np.max)
        power_to_db = (power_to_db + 80)/80

        return power_to_db
    def getChromaSTFT(self):
        logger.debug("")
        return librosa.feature.chroma_stft(y=self.y, sr=self.sr)
    def getMFCC(self):
        logger.debug("")
        return librosa.feature.mfcc(y=self.y, sr=self.sr)
    def getRMS(self):
        logger.debug("")
        S = librosa.feature.rms(y=self.y)
        return S
    def getCQT(self):
        logger.debug("")
        return librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
    def getCENS(self):
        logger.debug("")
        return librosa.feature.chroma_cens(y=self.y, sr=self.sr)
    def getCENT(self):
        logger.debug("")
        return librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
    def getBandWidth(self):
        logger.debug("")
        return librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)
    def getContrast(self):
        logger.debug("")
        S = np.abs(librosa.stft(self.y))
        return librosa.feature.spectral_contrast(S=S, sr=self.sr)
    def getFlatness(self):
        logger.debug("")
        return librosa.feature.spectral_flatness(y=self.y)
    def getRollOff(self):
        logger.debug("")
        return librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
    def getTonnetz(self):
        logger.debug("")
        temp_y = librosa.effects.harmonic(self.y)
        return librosa.feature.tonnetz(y=temp_y, sr=self.sr)
    def getZeroCrossingRate(self):
        logger.debug("")
        return librosa.feature.zero_crossing_rate(self.y)
    def getTempogram(self):
        logger.debug("")
        oenv = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        return librosa.feature.tempogram(onset_envelope=oenv, sr=self.sr)
    def getFourierTempogram(self):
        logger.debug("")
        oenv = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        return librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=self.sr)

class FeatureCollecter():
    def getFeaturesResampling(self, y, sr, increaseIndex=0, ):
        featureGetter = FeatureGetter(y,sr)
        sec = featureGetter.getLoudnessTimestamp()
        pivot = 1
        right = left = 5
        slice_y = featureGetter.getSlicedAudio(left,right,sec)
        slice_y = librosa.resample(slice_y, sr, 22050)
        
        # mel = librosa.feature.melspectrogram(y=slice_y, sr=tunedSR, hop_length=512, n_mels=128)
        mel = librosa.feature.melspectrogram(y=slice_y, sr=22050, hop_length=512, n_mels=128)

        timeseries_length = 431
        features = np.zeros((1, timeseries_length, 128), dtype=np.float64)
        features[0, :, 0:128] = mel.T[0:timeseries_length, :]

        return features

    def getEQFeature(self, y,sr):
        featureGetter = FeatureGetter(y,sr)
        defaultSize = 6000
        features = np.zeros((1, defaultSize, 128), dtype=np.float64)


        mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512, n_mels=128)
        amplitude_to_db = librosa.amplitude_to_db(mel,ref=1.0)

        # 6000 * 128           11281 * 128
        size = min(len(amplitude_to_db.T), defaultSize)
        # features[0, :, 0:128] = amplitude_to_db.T[0:size, :]
        features[0, 0:size, 0:128] = amplitude_to_db.T[0:size, :]
        return features

    def getMelBytimeShift(self, y, sr, increaseIndex=0, increaseGap=0.2):
        featureGetter = FeatureGetter(y,sr)
        sec = featureGetter.getLoudnessTimestamp()
        pivot = 1
        if increaseIndex % 2 == 0: pivot = -1
        left = 5 - ((int((increaseIndex + 1)/2) * increaseGap) * pivot)
        right = 5 + ((int((increaseIndex + 1)/2) * increaseGap) * pivot)
        
        slice_y = featureGetter.getSlicedAudio(left,right,sec)
        mel = librosa.feature.melspectrogram(y=slice_y, sr=sr, hop_length=512, n_mels=128)
        # mfcc = librosa.feature.mfcc(y=slice_y, sr=sr, n_mfcc=128)        
        return mel

    def getMelBychangeSpeed(self, y, sr, increaseIndex=0):
        featureGetter = FeatureGetter(y,sr)
        sec = featureGetter.getLoudnessTimestamp()
        logger.debug("loudness sec : "+str(sec))
        pivot = 1
        right = left = 5
        
        slice_y = featureGetter.getSlicedAudio(left,right,sec)

        if increaseIndex % 2 == 0: pivot = -1
        tunedSR = sr * (1+((int((increaseIndex+1)/2)*0.1)*pivot))

        mel = librosa.feature.melspectrogram(y=slice_y, sr=tunedSR, hop_length=512, n_mels=128)
        return mel
    
    def getMelBychnagePitch(self, y, sr, increaseIndex=0, increaseGap=0.2):
        featureGetter = FeatureGetter(y,sr)
        sec = featureGetter.getLoudnessTimestamp()
        pivot = 1
        right = left = 5
        
        slice_y = featureGetter.getSlicedAudio(left,right,sec)
        
        if increaseIndex % 2 == 0: pivot = -1
        tone = int((increaseIndex+1)/2) * pivot
        y_pitch = librosa.effects.pitch_shift(slice_y, sr, n_steps=tone)

        mel = librosa.feature.melspectrogram(y=y_pitch, sr=sr, hop_length=512, n_mels=128)
        return mel

    def getMelByNoise(self, y, sr, increaseIndex=0):
        featureGetter = FeatureGetter(y,sr)
        sec = featureGetter.getLoudnessTimestamp()
        pivot = 1
        right = left = 5
        
        slice_y = featureGetter.getSlicedAudio(left,right,sec)
        noise = np.random.randn(len(slice_y))
        if increaseIndex != 0 :
            slice_y = slice_y + 0.005 * noise
            slice_y = slice_y.astype(type(y[0]))

        mel = librosa.feature.melspectrogram(y=slice_y, sr=sr, hop_length=512, n_mels=128)
        return mel
        

    def getFeatures(self, y, sr, increaseIndex=0, increaseGap=0.2):
        
        if increaseIndex % 3 == 0:
            mel = self.getMelBychangeSpeed(y, sr, increaseIndex)
        elif increaseIndex % 3 == 1:
            mel = self.getMelBytimeShift(y, sr, increaseIndex, increaseGap)
        # elif increaseIndex % 4 == 2:
        #     mel = self.getMelBychnagePitch(y, sr, increaseIndex, increaseGap)
        else:
            mel = self.getMelByNoise(y, sr, increaseIndex)

        # mel = self.getMelBychnagePitch(y, sr, increaseIndex, increaseGap)
        # amplitude_to_db = librosa.amplitude_to_db(mel,ref=1.0)
        
        timeseries_length = 431
        features = np.zeros((1, timeseries_length, 128), dtype=np.float64)
        features[0, :, 0:128] = mel.T[0:timeseries_length, :]

        return features