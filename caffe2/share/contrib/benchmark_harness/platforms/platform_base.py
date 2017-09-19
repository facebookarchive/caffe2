#!/usr/bin/env python3

from arg_parse import getArgs

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

    def runOnPlatform(self):
        self.setupPlatform()
        self.runBenchmark()
        return self.collectData()

    def collectData(self):
        results = []
        rows = self.output.split('\n')
        useful_rows = [row for row in rows if row.find(self.IDENTIFIER) >= 0]
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
                    result[pair[0].strip()] = pair[1].strip()
                if useful_rows[i+1].find(self.OPERATOR_DELAYS_START) >= 0:
                    i = self._collectOperatorDelayData(useful_rows, result, i+1)
                results.append(result)
            i += 1
        assert len(results) == getArgs().iter, "Incorrect number of results collected"
        return results


    def _collectOperatorDelayData(self, rows, result, start_idx):
        res = {}
        i = start_idx+1
        while i < len(rows) and rows[i].find(self.OPERATOR_DELAYS_END) < 0:
            row = rows[i]
            start_idx = row.find(self.IDENTIFIER) + len(self.IDENTIFIER)
            pair = row[start_idx:].strip().split('-')
            assert len(pair) == 2, "Operator delay doesn't have two items"
            res[pair[0].strip()] = pair[1].strip()
            i = i+1
        result['Operators Delay'] = res
        return i
