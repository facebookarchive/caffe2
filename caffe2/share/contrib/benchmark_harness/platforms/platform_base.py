#!/usr/bin/env python3

class PlatformBase(object):
    # Class constant, need to change observer_reporter_print.cc if the names are changed
    IDENTIFIER = 'Caffe2Observer : '
    NET_NAME = 'Net Name'
    NET_DELAY = 'Net Delay'
    OPERATOR_DELAYS_START = 'Operators Delay Start'
    OPERATOR_DELAYS_END = 'Operators Delay End'
    def __init__(self):
        pass

    def setupPlatform(self):
        pass

    def runBenchmark(self):
        pass

    def collectData(self):
        pass

    def runOnPlatform(self):
        self.setupPlatform()
        self.runBenchmark()
        return self.collectData()
