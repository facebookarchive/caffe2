#!/usr/bin/env python

import argparse

args = None
parser = argparse.ArgumentParser(description="Perform one benchmark run")


def getParser():
    return parser

def parse():
    global args
    args = parser.parse_args()
    return args

def getArgs():
    return args
