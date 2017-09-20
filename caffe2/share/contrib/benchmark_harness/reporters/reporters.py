#!/usr/bin/env python

from utils.arg_parse import getArgs
from local_reporter.local_reporter import LocalReporter


def getReporters():
    reporters = []
    if getArgs().local_reporter:
        reporters.append(LocalReporter())
    return reporters
