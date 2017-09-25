#!/usr/bin/env python

from utils.arg_parse import getArgs
from local_reporter.local_reporter import LocalReporter
from remote_reporter.remote_reporter import RemoteReporter

def getReporters():
    reporters = []
    if getArgs().local_reporter:
        reporters.append(LocalReporter())
    if getArgs().remote_reporter:
        reporters.append(RemoteReporter())
    return reporters
