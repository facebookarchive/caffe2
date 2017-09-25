#!/usr/bin/env python

import argparse

args = None
unknowns = []
parser = argparse.ArgumentParser(description="Perform one benchmark run")


def getParser():
    return parser

def parse(with_unknowns=False):
    global args, unknowns
    if with_unknowns:
        args, unknowns = parser.parse_known_args()
    else:
        args = parser.parse_args()
    return args

def getArgs():
    return args

def getUnknowns():
    return unknowns
