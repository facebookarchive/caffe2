#!/usr/bin/env python3

import argparse

args = None

parser = argparse.ArgumentParser(description="Perform one benchmark run")
parser.add_argument("--net", required=True,
    help="The given predict net to benchmark.")
parser.add_argument("--init_net", required=True,
    help="The given net to initialize any parameters.")
parser.add_argument("--input",
    help="Input that is needed for running the network. "
    "If multiple input needed, use comma separated string.")
parser.add_argument("--input_file",
    help="Input file that contain the serialized protobuf for "
    "the input blobs. If multiple input needed, use comma "
    "separated string. Must have the same number of items "
    "as input does.")
parser.add_argument("--input_dims",
    help="Alternate to input_files, if all inputs are simple "
    "float TensorCPUs, specify the dimension using comma "
    "separated numbers. If multiple input needed, use "
    "semicolon to separate the dimension of different "
    "tensors.")
parser.add_argument("--output",
    help="Output that should be dumped after the execution "
    "finishes. If multiple outputs are needed, use comma "
    "separated string. If you want to dump everything, pass "
    "'*' as the output value.")
parser.add_argument("--output_folder",
    help="The folder that the output should be written to. This "
    "folder must already exist in the file system.")
parser.add_argument("--warmup", default=0, type=int,
    help="The number of iterations to warm up.")
parser.add_argument("--iter", default=10, type=int,
    help="The number of iterations to run.")
parser.add_argument("--run_individual", action="store_true",
    help="Whether to benchmark individual operators.")
parser.add_argument("program", help="The program to run on the platform.")

def getParser():
    return parser

def parse():
    global args
    args = parser.parse_args()
    return args

def getArgs():
    return args
