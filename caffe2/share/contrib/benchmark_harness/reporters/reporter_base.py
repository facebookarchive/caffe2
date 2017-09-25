#!/usr/bin/env python

class ReporterBase(object):
    DATA = 'data'
    META = 'meta'
    NET_NAME = 'Net Name'
    PLATFORM = 'platform'
    def __init__(self):
        pass

    def report(self, content):
        pass
