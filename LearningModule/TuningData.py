
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Util.logger import logger

class TuningData:
    def __init__(self):
        self.data = None
        self.gain = self.ratio = self.release = self.threshold = self.attack = None
    def __init__(self, arr):
        self.data = arr
        self.threshold = arr[0]
        self.ratio = arr[1]
        self.attack = arr[2]
        self.release = arr[3]
        self.gain = arr[4]

    def setThreshold(self, data):
        self.threshold = data
    def setAttack(self, data):
        self.attack = max(data, 0)
    def setRelease(self, data):
        self.release = max(data, 0)
    def setRatio(self, data):
        self.ratio = max(data, 0)
    def setGain(self, data):
        self.gain = data

    def normalize(self):
        # labels = [[-6.9, 1.4, 78, 40, 6.7]]
        # 0 ~ -10
        self.data[0] = self.threshold = self.threshold * -0.1
        # 1 ~ 2
        self.data[1] = self.ratio = self.ratio - 1
        # 0 ~ 100
        self.data[2] = self.attack = self.attack * 0.01
        self.data[3] = self.release = self.release * 0.01
        # 0 ~ 10
        self.data[4] = self.gain = self.gain * 0.1

    def unnormalize(self):
        self.data[0] = self.threshold = min(self.threshold * -10, 0)
        self.data[1] = self.ratio = max(self.ratio + 1, 0)
        self.data[2] = self.attack = max(self.attack * 100, 0)
        self.data[3] = self.release = max(self.release * 100, 0)
        self.data[4] = self.gain = self.gain * 10

    def printData(self):
        logger.debug("Threshold : " + str(self.threshold))
        logger.debug("Ratio : " + str(self.ratio))
        logger.debug("Attack : " + str(self.attack))
        logger.debug("Release : " + str(self.release))
        logger.debug("Gain : " + str(self.gain))
