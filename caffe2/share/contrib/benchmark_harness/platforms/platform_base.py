#!/usr/bin/env python3

class PlatformBase(object):
    def __init__(self, args):
        self.args = args
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
        self.collectData()
