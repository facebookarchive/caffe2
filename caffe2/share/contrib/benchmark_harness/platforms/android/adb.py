#!/usr/bin/env python

import subprocess
import os.path as path
from utils.custom_logger import getLogger

class ADB(object):
    def __init__(self, device=None):
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

    def logcat(self, *args):
        return self.run("logcat", *args)

    def shell(self, cmd):
        return self.run("shell", cmd)

    def _call(self, subprocess_cmd, cmd, **kwargs):
        getLogger().info('Running: %s', ' '.join(cmd))
        return subprocess_cmd(cmd, **kwargs)
