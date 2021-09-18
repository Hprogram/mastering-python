import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import wave
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
from FeatureExtract import Normalizer

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

if (__name__ == '__main__'):
    #path = "FeatureExtract/22k_except/pop_voc_2.wav"
    # path = "FeatureExtract/LearnData/hiphop/h_voc_6.wav"
    # path = "FeatureExtract/LearnData/pop/pop_voc_3.wav"
    # path = "Test/after_pop_38.wav"
    # path = "FeatureExtract/LearnData_22k_mono/hiphop/h_66.wav"
    path = "music/pop_voc_2.wav"
    f = open(path, 'rb')
    # f = open("FeatureExtract/LearnData_22k_mono/hiphop/h_66.wav", 'rb')
    # voc1 -6.9 1.4 78 40 6.7
    # rnb1 -5.4 1.5 69 34 8.7
    # rnb2 -5.1 1.5 79 41 7.9

    data = f.read()
    f.close()

    origin_data, samplerate = sf.read(file=io.BytesIO(data), dtype='float32')
    a,b = sf.read(file=io.BytesIO(data), dtype='float32')

    logger.debug("Start Transform to Mono")
    transform_data = librosa.to_mono(origin_data.T)
    # transform_data = librosa.resample(transform_data, samplerate, 22050)
    featureCollecter = FeatureGetter.FeatureCollecter()

    logger.debug("Start GetFeatures")

    # features = featureCollecter.getFeatures(transform_data, 22050,0)
    features = featureCollecter.getFeaturesResampling(transform_data,samplerate,0)
    features = features.reshape(1,431,128,1)
    # features = ""
    logger.debug("start DeepLearning")

    # CNN
    model = tf.keras.models.load_model('./Result/132.3_05-02-14-29-35_(256, 256, 256, 256, 256, 256, 256)_(3, 3, 3, 5, 5, 5, 5)_1320.h5')
    # model = tf.keras.models.load_model('134.9_05-06-17-09-44_(32, 64, 256, 256, 256, 256)_(7, 7, 7, 7, 7, 7)_2318.h5')
    # model = tf.keras.models.load_model("_0.013367678038775921('relu', 'sigmoid', 'sigmoid')('adam', 'mse').h5")
    answer = model.predict(features)

    # model = tf.keras.models.load_model("_0.010390196815133096('sigmoid', 'relu', 'sigmoid')('adam', 'mse').h5")
    # answer = model.predict(features.reshape(1,features.shape[0]*features.shape[1]))
    
    logger.debug(answer)
    normalizer = Normalizer.Normalizer()
    
    logger.debug("audio path : " + path)
    logger.debug(normalizer.unStandardization(answer))


# TEST
    compAnswer = answer;
    model = tf.keras.models.load_model('./ResultEQ/142.9_05-25-16-18-29_(32, 64, 128, 128, 128, 256)_(3, 5, 5, 7, 7, 7)_1559.h5')

    # features = featureCollecter.getEQFeature(transform_data, 22050)
    # features = features.reshape(features.shape[0],features.shape[1],features.shape[2],1)
    EQAnswer = model.predict(features)
    from FeatureExtract import EQNormalizer
    eqNormalizer = EQNormalizer.EQNormalizer()
    logger.debug(eqNormalizer.unStandardization(EQAnswer))

# Tuning
    tuningData = TuningData.TuningData(compAnswer.reshape(compAnswer.shape[1]))

    tuningModule2 = TuningModule(tuningData, origin_data.T)
    # masteredAudio = tuningModule2.getTunedAudio()
    # path = path.replace(".wav", "_comp.wav")
    # f = open(path, 'wb')
    # sf.write(f, masteredAudio.T, samplerate, format='WAV')
    # f.close()

    masteredAudio = tuningModule2.getTunedAudioCOMPEQ(EQAnswer.reshape(EQAnswer.shape[1]))
    # path = path.replace("_comp.wav", "_comp_eq.wav")
    path = path.replace(".wav", "_comp_eq.wav")
    f = open(path, 'wb')
    sf.write(f, masteredAudio.T, samplerate, format='WAV')
    f.close()
    logger.debug("done")
