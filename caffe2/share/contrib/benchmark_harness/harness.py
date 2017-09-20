#!/usr/bin/env python

import sys

import utils.arg_parse
from platforms.platforms import getPlatforms
from reporters.reporters import getReporters
from utils.arg_parse import getParser

getParser().add_argument("--net", required=True,
    help="The given predict net to benchmark.")
getParser().add_argument("--init_net", required=True,
    help="The given net to initialize any parameters.")
getParser().add_argument("--input",
    help="Input that is needed for running the network. "
    "If multiple input needed, use comma separated string.")
getParser().add_argument("--input_file",
    help="Input file that contain the serialized protobuf for "
    "the input blobs. If multiple input needed, use comma "
    "separated string. Must have the same number of items "
    "as input does.")
getParser().add_argument("--input_dims",
    help="Alternate to input_files, if all inputs are simple "
    "float TensorCPUs, specify the dimension using comma "
    "separated numbers. If multiple input needed, use "
    "semicolon to separate the dimension of different "
    "tensors.")
getParser().add_argument("--output",
    help="Output that should be dumped after the execution "
    "finishes. If multiple outputs are needed, use comma "
    "separated string. If you want to dump everything, pass "
    "'*' as the output value.")
getParser().add_argument("--output_folder",
    help="The folder that the output should be written to. This "
    "folder must already exist in the file system.")
getParser().add_argument("--warmup", default=0, type=int,
    help="The number of iterations to warm up.")
getParser().add_argument("--iter", default=10, type=int,
    help="The number of iterations to run.")
getParser().add_argument("--run_individual", action="store_true",
    help="Whether to benchmark individual operators.")
getParser().add_argument("program", help="The program to run on the platform.")
getParser().add_argument("--git_commit",
    help="The git commit on this benchmark run.")

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
