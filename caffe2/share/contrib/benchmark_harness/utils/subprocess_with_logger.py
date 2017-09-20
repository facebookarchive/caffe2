#!/usr/bin/env python

import subprocess
from utils.custom_logger import getLogger

def processRun(*args):
    getLogger().info("Running: %s", ' '.join(*args))
    return subprocess.check_output(*args)
