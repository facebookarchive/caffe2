#!/usr/bin/env python3

import logging
logging.basicConfig()

logger = logging.getLogger("GlobalLogger")

logger.setLevel(logging.DEBUG)

def getLogger():
    return logger
