import zerorpc
from Util.logger import *
import numpy as np
if (__name__ == '__main__'):
    temp = np.zeros((908,10000), dtype=np.float32)
    
    logger.debug("debug start")
    c = zerorpc.Client(timeout=60, heartbeat=5)
    c.connect("tcp://127.0.0.1:4242")

    f = open("../backend/routes/sample_test_eq.wav", 'rb')
    data = f.read()
    f.close()

    masteredAudioBuffer = c.getMasteredAudio(data)
    a = 1
    a = a+1