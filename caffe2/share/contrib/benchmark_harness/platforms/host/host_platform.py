#!/usr/bin/env python3

from platforms.platform_base import PlatformBase
import logging
import subprocess
from arg_parse import getArgs, getParser

logger = logging.getLogger(__name__)

getParser().add_argument("--host", action="store_true",
    help="Run the benchmark on the host.")

class HostPlatform(PlatformBase):
    def __init__(self):
        super(HostPlatform, self).__init__()
        self.output = None

    def setupPlatform(self):
        pass

    def runBenchmark(self):
        cmd = [
            getArgs().program,
            "--logtostderr", "1",
            "--init_net", getArgs().init_net,
            "--net", getArgs().net,
            "--input", getArgs().input,
            "--warmup", str(getArgs().warmup),
            "--iter", str(getArgs().iter),
            ]
        if getArgs().input_file:
            cmd.extend(["--input_file", getArgs().input_file])
        if getArgs().input_dims:
            cmd.extend(["--input_dims", getArgs().input_dims])
        if getArgs().output:
            cmd.extend(["--output", getArgs().output])
            cmd.extend(["--output_folder", getArgs().output_folder + "output"])
        if getArgs().run_individual:
            cmd.extend(["--run_individual", "true"])
        command = ' '.join(cmd)
        logger.log(logging.INFO,
                    "Running: %s",
                    command)
        pipes = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        std_out, std_err = pipes.communicate()
        assert pipes.returncode == 0, "Benchmark run failed"
        if len(std_err):
            self.output = std_err

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
