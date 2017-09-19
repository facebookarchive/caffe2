#!/usr/bin/env python3

import sys

import arg_parse
from platforms.platforms import getPlatforms
from reporters.reporters import getReporters
import logging

logger = logging.getLogger(__name__)

class BenchmarkDriver(object):
    def __init__(self):
        self.platforms = []
        arg_parse.parse()

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
