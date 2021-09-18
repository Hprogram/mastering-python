import soundfile as sf
import io
import librosa
import numpy as np
import datetime
from FeatureExtract import FeatureGetter
from TuningModule import *
from LearningModule import TuningData
from Util.logger import *          
import tensorflow as tf

from moviepy.editor import *
from moviepy.audio.AudioClip import AudioArrayClip
import random
import string

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
session = InteractiveSession(config=config)

class MasteringRPC(object):
    def __init__(self):
        logger.debug("init")
        self.model = tf.keras.models.load_model('./Result/120.7_11-17-21-44-54_(256, 256, 256, 256, 256, 256, 256)_(3, 3, 3, 5, 5, 5, 5)_155_init.h5')
        logger.debug("init step 1.1")
        self.eqModel = tf.keras.models.load_model('./ResultEQ/142.9_05-25-16-18-29_(32, 64, 128, 128, 128, 256)_(3, 5, 5, 7, 7, 7)_1559.h5')
        logger.debug("init step 2")

    def predictTuningData(self, features):
        answer = self.model.predict(features)
        from FeatureExtract import Normalizer
        normalizer = Normalizer.Normalizer()
        normalizer.unStandardization(answer)
        logger.debug(answer)
        tuningData = TuningData.TuningData(answer.reshape(answer.shape[1]))
        tuningData.printData()
        return tuningData

    def getMasteredAudioFromY(self, y, sr):
        origin_data = y
        samplerate = sr
        logger.debug("Start Transform to Mono")
        transform_data = librosa.to_mono(origin_data.T)

        featureCollecter = FeatureGetter.FeatureCollecter()
        logger.debug("Start GetFeatures")
        features = featureCollecter.getFeaturesResampling(transform_data,samplerate,0)
        features = features.reshape(1,431,128,1)
        # features = ""
        logger.debug("start DeepLearning")
        tuningData = self.predictTuningData(features)

        logger.debug("Predict EQ Before");
        EQAnswer = self.eqModel.predict(features)
        logger.debug("Predict EQ After");
        from FeatureExtract import EQNormalizer
        eqNormalizer = EQNormalizer.EQNormalizer()
        logger.debug(eqNormalizer.unStandardization(EQAnswer))


        tuningModule = TuningModule(tuningData, origin_data.T)
        logger.debug("Start Tuning")
        masteredAudio = tuningModule.getTunedAudioCOMPEQ(EQAnswer.reshape(EQAnswer.shape[1]))
        targetfile = io.BytesIO()
        logger.debug("Start Write Mastered Audio Buffer")
        sf.write(targetfile, masteredAudio.T, samplerate, format='WAV')
        
        # Test Code Start Make Mixed Audio (Mastered + Origin)
        # mastered_T = masteredAudio
        # origin_data_T = origin_data.T
        # mastered_T[1] = mastered_T[0]
        # mastered_T[0] = origin_data_T[0]
        # logger.debug("test")
        # sf.write("./sample1.wav", mastered_T.T, samplerate, format="WAV")
        # Test Code End

        return targetfile.getvalue()

    def getMasteredAudio(self, buffer, fileName):
        if '.wav' not in fileName: 
            return self.getMasteredVideo(buffer, fileName)
        origin_data, samplerate = sf.read(file=io.BytesIO(buffer), dtype='float32')
        return self.getMasteredAudioFromY(origin_data, samplerate)


    def _temp_file_name(self, ext):
        return ("./tmp-video-"
            + "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            + ext)
    def getAudioFromVideo(self, data):
        temp_file = self._temp_file_name(ext="mp4")
        with open(temp_file, "wb") as file:
            file.write(data)
        videoclip = VideoFileClip(temp_file)
        # os.remove(temp_file)
        audioclip = videoclip.audio
        # audioclip.write_audiofile("./test_origin.wav")

        return audioclip.to_soundarray(), audioclip.fps, videoclip, temp_file
    def getMasteredVideo(self, videoBuffer, fileName):
        y, sr, originVideoClip, tmpFileName = self.getAudioFromVideo(videoBuffer)
        masteredAudio = self.getMasteredAudioFromY(y,sr)
        mastered_y, samplerate = sf.read(file=io.BytesIO(masteredAudio), dtype='float32')
        
        
        mastered_audioclip = AudioArrayClip(mastered_y, fps=sr)
        # mastered_audioclip.write_audiofile("./test.wav")

        masteredVideoClip = originVideoClip.set_audio(mastered_audioclip)
        
        masteredTempFileName = self._temp_file_name(os.path.splitext(fileName)[1])
        masteredVideoClip.write_videofile("./" + masteredTempFileName)

        path = "./" + masteredTempFileName
        f = open(path, 'rb')
        data = f.read()
        f.close()

        os.remove("./" + masteredTempFileName)
        os.remove(tmpFileName)

        return data


