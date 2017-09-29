#!/usr/bin/env python

from utils.subprocess_with_logger import processRun
from utils.custom_logger import getLogger

class Git(object):
    def __init__(self, dir):
        self.dir = dir
        pass

    def run(self, cmd, *args):
        git = ["git"]
        if self.dir:
            git.append("-C")
            git.append(self.dir)
        git.append(cmd)
        git.extend(args)
        return processRun(git)

    def pull(self, *args):
        return self.run('pull', *args)

    def checkout(self, *args):
        return self.run('checkout', *args)

    def getCommitHash(self, commit):
        return self.run('rev-parse', commit).rstrip()

    def getCommitTime(self, commit):
        return int(self.run('show', '-s', '--format=%ct', commit).strip())
