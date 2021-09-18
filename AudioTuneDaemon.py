import zerorpc
from Util.logger import *
from MasteringRPC import *          

logger.debug("start server")
s = zerorpc.Server(MasteringRPC())
s.bind("tcp://0.0.0.0:4242")
s.run()
print(1)