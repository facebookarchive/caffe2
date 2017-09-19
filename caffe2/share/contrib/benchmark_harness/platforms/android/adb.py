#!/usr/bin/env python3

import logging
import subprocess
import os.path as path

logger = logging.getLogger(__name__)

class ADB(object):
    def __init__(self, device=None, adb_log_level=logging.INFO):
        self.adb_log_level = adb_log_level
        self.device = device
        self.dir = "/data/local/tmp/"

    def run(self, cmd, *args):
        adb = ["adb"]
        if self.device:
            adb.append("-s")
            adb.append(self.device)
        adb.append(cmd)
        for item in args:
            if isinstance(item, list):
                adb.extend(item)
            else:
                adb.append(item)
        return self._call(subprocess.check_output, adb)

    def push(self, src, tgt = None):
        if tgt is None:
            tgt = src
        basename = path.basename(tgt)
        target = self.dir + basename
        return self.run("push", src, target)

    def pull(self, src, tgt):
        basename = path.basename(src)
        source = self.dir + basename
        return self.run("pull", source, tgt)

    def shell(self, cmd):
        return self.run("shell", cmd)

    def _call(self, subprocess_cmd, cmd, **kwargs):
        logger.log(self.adb_log_level,
                   'Running: %s',
                   ' '.join(cmd))
        return subprocess_cmd(cmd, **kwargs)
