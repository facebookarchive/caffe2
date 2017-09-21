#!/usr/bin/env python

import collections
import time
from utils.arg_parse import getArgs

class PlatformBase(object):
    # Class constant, need to change observer_reporter_print.cc if the names are changed
    IDENTIFIER = 'Caffe2Observer : '
    NET_NAME = 'Net Name'
    NET_DELAY = 'Net Delay'
    OPERATOR_DELAYS_START = 'Operators Delay Start'
    OPERATOR_DELAYS_END = 'Operators Delay End'
    DETAILS = 'details'
    SUMMARY = 'summary'
    PLATFORM = 'platform'
    COMMIT = 'commit'
    def __init__(self):
        self.output = None
        pass

    def setupPlatform(self):
        pass

    def runBenchmark(self):
        pass

    def runOnPlatform(self):
        self.setupPlatform()
        self.runBenchmark()
        return self.collectData()

    def getOutput(self):
        return self.output

    def collectData(self):
        results = []
        rows = self.output.split('\n')
        useful_rows = [row for row in rows if row.find(self.IDENTIFIER) >= 0]
        net_name = None
        i = 0
        while (i < len(useful_rows)):
            row = useful_rows[i]
            net_start_idx = row.find(self.NET_NAME)
            if net_start_idx > 0:
                content = row[net_start_idx:].split(':')
                assert len(content) == 2, "Net delay row doesn't have two items"
                result = {}
                for x in content:
                    pair = x.strip().split('-')
                    assert len(pair) == 2, "Net delay field doesn't have two items"
                    if pair[0].strip() == self.NET_NAME:
                        net_name = pair[1].strip()
                    else:
                        result[pair[0].strip()] = float(pair[1].strip())
                if useful_rows[i+1].find(self.OPERATOR_DELAYS_START) >= 0:
                    i = self._collectOperatorDelayData(useful_rows, result, i+1)
                results.append(result)
            i += 1
        assert len(results) == getArgs().iter, "Incorrect number of results collected"
        return self._processData(results, net_name)


    def _collectOperatorDelayData(self, rows, result, start_idx):
        i = start_idx+1
        while i < len(rows) and rows[i].find(self.OPERATOR_DELAYS_END) < 0:
            row = rows[i]
            start_idx = row.find(self.IDENTIFIER) + len(self.IDENTIFIER)
            pair = row[start_idx:].strip().split('-')
            assert len(pair) == 2, "Operator delay doesn't have two items"
            result[pair[0].strip()] = float(pair[1].strip())
            i = i+1
        return i

    def _processData(self, data, net_name):
        ts = time.time()
        details = collections.defaultdict(list)
        for d in data:
            for k, v in d.items():
                details[k].append(v)
        for d in details:
            details[d].sort()

        summary = {}
        summary['time'] = ts
        summary[self.NET_NAME] = net_name
        for d in details:
            values = details[d]
            length = len(values)
            one_summary = {}
            one_summary['min'] = values[0]
            one_summary['max'] = values[-1]
            if (len(d) % 2) == 1:
                one_summary['median'] = values[length // 2]
            else:
                one_summary['median'] = (values[(length - 1) //2] + values[length // 2]) / 2
            summary[d] = one_summary
        if getArgs().git_commit:
            summary[self.COMMIT] = getArgs().git_commit
        results = {}
        results[self.DETAILS] = details
        results[self.SUMMARY] = summary
        return results
