#!/usr/bin/env python

import sys

import utils.arg_parse
from platforms.platforms import getPlatforms
from reporters.reporters import getReporters

class BenchmarkDriver(object):
    def __init__(self):
        self.platforms = []
        utils.arg_parse.parse()

    def runBenchmark(self, platforms):
        reporters = getReporters()
        for platform in platforms:
            data = platform.runOnPlatform()
            for reporter in reporters:
                reporter.report(data)

    def run(self):
        platforms = getPlatforms()
        self.runBenchmark(platforms)

if __name__ == "__main__":
    app = BenchmarkDriver()
    app.run()
