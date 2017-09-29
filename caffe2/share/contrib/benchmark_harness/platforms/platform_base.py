#!/usr/bin/env python

import collections
import time
import sys
from utils.arg_parse import getArgs

class PlatformBase(object):
    # Class constant, need to change observer_reporter_print.cc if the names are changed
    IDENTIFIER = 'Caffe2Observer : '
    NET_NAME = 'Net Name'
    NET_DELAY = 'Net Delay'
    OPERATOR_DELAYS_START = 'Operators Delay Start'
    OPERATOR_DELAYS_END = 'Operators Delay End'
    DATA = 'data'
    META = 'meta'
    PLATFORM = 'platform'
    COMMIT = 'commit'
    def __init__(self):
        self.info = self._processInfo()
        pass

    def setupPlatform(self, info):
        pass

    def runBenchmark(self, info):
        return None

    def runOnPlatform(self):
        assert self.git_info and self.git_info['treatment'],
            "Treatment is not specified."
        # Run on the treatment
        treatment_info = self.info['treatment']
        result = {}
        result['treatment'] = self.runOneIter(treatment_info)

        # Run on the control
        if self.info['control']:
            # wait till the device is cooler
            timne.sleep(60)
            control_info = self.info['control']
            result['control'] = self.runOneIter(control_info)
        return self._mergeResult(result)

    def runOneIter(self, one_info):
        self.setupPlatform(one_info)
        outout = self.runBenchmark(one_info)
        return self.collectData(one_info, output)

    def collectData(self, info, output):
        results = []
        rows = output.split('\n')
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
        if len(results) > getArgs().iter:
            # Android 5 has an issue that logcat -c does not clear the entry
            results = results[-getArgs().iter:]
        else:
            assert len(results) == getArgs().iter, "Incorrect number of results collected"
        return self._processData(info, results, net_name)


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

    def _processData(self, info, data, net_name):
        ts = time.time()
        details = collections.defaultdict(list)
        for d in data:
            for k, v in d.items():
                details[k].append(v)
        for d in details:
            details[d].sort()

        processed_data = {}
        for d in details:
            values = details[d]
            assert len(values) > 0
            processed_data[d] = {
                'values' : values,
                'summary' : {
                    'min' : values[0],
                    'max' : values[-1],
                    'median' : self._getMedian(values),
                }
            }
        meta = {}
        meta['time'] = ts
        meta[self.NET_NAME] = net_name
        meta['command'] = ' '.join(sys.argv)
        if info['commit']:
            meta[self.COMMIT] = info['commit']
        if info['commit_time']:
            meta['commit_time'] = info['commit_time']
        results = {}
        results[self.DATA] = processed_data
        results[self.META] = meta
        return results

    def _getMedian(self, values):
        length = len(values)
        return values[length // 2] if (length % 2) == 1 else \
            (values[(length - 1) //2] + values[length // 2]) / 2


    def _processInfo(self):
        if getArgs().program:
            assert not getArgs().android and getArgs().host,
                "Cannot specify both --android and --host when --program is specified."
            git_info = {'treatment' : {}}
            if getArgs().android:
                git_info['treatment']['program_android'] = getArgs().program
            elif getArgs.host:
                git_info['treatment']['program_host'] = getArgs().program
            return git_info
        elif getArgs.info:
            return json.loads(getArgs().info)
        else:
            assert False,
                "Must specify either --program or --git_info in command line"

    def _mergeResult(self, result):
        if not result['control']:
            return result
        treatment = result['treatment']
        control = result['control']
        data = {}
        data[self.DATA] = {}
        data[self.META] = treatment[self.META]
        for k in treatment[self.DATA]:
            treatment_value = treatment[self.DATA][k]
            control_value = control[self.DATA][k]
            assert control_value,
                "Value %s existed in treatment but not control", k
            data[self.DATA][k] = treatment_value
            for control_key in control_value:
                new_key = 'control_' + control_key
                data[self.DATA][k][new_key] = control_value[control_key]
        data[self.META]['control_time'] = control[self.META]['time']
        data[self.META]['control_commit'] = control[self.META]['commit']
        data[self.META]['control_commit_time'] = control[self.META]['commit_time']
        return data
