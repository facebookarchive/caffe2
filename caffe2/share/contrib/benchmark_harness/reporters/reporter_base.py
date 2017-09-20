#!/usr/bin/env python3

class ReporterBase(object):
    DETAILS = 'details'
    SUMMARY = 'summary'
    NET_NAME = 'Net Name'    
    def __init__(self):
        pass

    def report(self, content):
        pass
