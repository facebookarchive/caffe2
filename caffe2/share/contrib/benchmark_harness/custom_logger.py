#!/usr/bin/env python

import logging
logging.basicConfig()

logger = logging.getLogger("GlobalLogger")

logger.setLevel(logging.DEBUG)

def getLogger():
    return logger
