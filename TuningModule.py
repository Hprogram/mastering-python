import librosa
from pysndfx.dsp import AudioEffectsChain
from LearningModule import TuningData
import soundfile as sf
import io

# import logging
# logger = logging.getLogger('pysndfx')
# logger.setLevel(logging.DEBUG)
class TuningModule:
    def __init__(self, tuningValue, originPath, targetPath):
        self.tuningValue = tuningValue
        self.originPath = originPath
        self.targetPath = targetPath
    def __init__(self, tuningValue, originAudioData):
        self.tuningValue = tuningValue
        self.originAudioData = originAudioData

    def appendEQ(self, audio_effecter, eqInfo):
        # audio_effecter.highpass(25)
        # audio_effecter.highshelf(1, 10000)
        # audio_effecter.equalizer(80, db=-1, q=3)
        # audio_effecter.equalizer(200, db=0.7, q =3)
        # audio_effecter.equalizer(1100, db=0.5, q=3)
        audio_effecter.highpass(eqInfo[0])
        audio_effecter.highshelf(eqInfo[2], eqInfo[1])
        audio_effecter.equalizer(eqInfo[3], db=eqInfo[4], q=3)
        audio_effecter.equalizer(eqInfo[5], db=eqInfo[6], q =3)
        audio_effecter.equalizer(eqInfo[7], db=eqInfo[8], q=3)

    def appendComp(self, audio_effecter):
        audio_effecter.custom('compand')
        audio_effecter.custom(str(self.tuningValue.attack*0.001) + ',' + str(self.tuningValue.release*0.001))
        db_from = 0
        db_to = self.tuningValue.threshold - (self.tuningValue.threshold / self.tuningValue.ratio)
        audio_effecter.custom(str(self.tuningValue.threshold) + ',' + str(db_from) + ',' + str(db_to) + ' 0 -60')

    def appendGain(self, audio_effecter):
        audio_effecter.gain(db=self.tuningValue.gain)

    def appendLimiter(self, audio_effecter):
        audio_effecter.custom('compand')
        audio_effecter.custom('0,0')
        audio_effecter.custom('-0.1,0,-0.1 0 -60 0')
    def getTunedAudioCOMPEQ(self, eqInfo):
        audio_effecter = AudioEffectsChain()
        self.appendEQ(audio_effecter, eqInfo)
        self.appendComp(audio_effecter)
        self.appendGain(audio_effecter)
        self.appendLimiter(audio_effecter)
        return audio_effecter(self.originAudioData);
    def getTunedAudio(self):
        audio_effecter = AudioEffectsChain()
        # self.appendEQ(audio_effecter)
        self.appendComp(audio_effecter)
        self.appendGain(audio_effecter)
        self.appendLimiter(audio_effecter)
        return audio_effecter(self.originAudioData);

if (__name__ == '__main__'):
    data = [-5.7,1.2,69,40,8.3]
    tuningData = TuningData.TuningData(data)

    # f = open("./sample/The Beatles---Let It Be .wav", 'rb')
    f = open("Test/pop_38.wav",'rb')
    data = f.read()
    f.close()

    y, sr = sf.read(file=io.BytesIO(data), dtype='float32')
    tuningModule2 = TuningModule(tuningData, y.T)
    masteredAudio = tuningModule2.getTunedAudio()

    f = open("./testssss.wav", 'wb')
    sf.write(f, masteredAudio.T, sr, format='WAV')
    f.close()