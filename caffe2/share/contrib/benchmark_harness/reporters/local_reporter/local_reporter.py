#!/usr/bin/env python

from utils.arg_parse import getParser, getArgs
from reporters.reporter_base import ReporterBase

import json
import os
import shutil
import datetime

getParser().add_argument("--local_reporter",
    help="Save the result to a directory specified by this argument.")


class LocalReporter(ReporterBase):
    def __init__(self):
        super(LocalReporter, self).__init__()

    def report(self, content):
        net_name = content[self.META][self.NET_NAME]
        netdir = self._getFilename(net_name) + "/"
        platform_name = content[self.META][self.PLATFORM]
        platformdir = self._getFilename(platform_name) + "/"
        ts = float(content[self.META]['time'])
        dt = datetime.datetime.fromtimestamp(ts)
        datedir = str(dt.year) + "/" + str(dt.month) + "/" + str(dt.day) + "/"
        dirname = platformdir + netdir + datedir
        dirname = getArgs().local_reporter + "/" + dirname
        i = 0
        while os.path.exists(dirname + str(i)):
            i = i+1
        dirname = dirname + str(i) + "/"
        os.makedirs(dirname)
        data = content[self.DATA]
        for d in data:
            filename = dirname + self._getFilename(d) + ".txt"
            content_d = json.dumps(data[d])
            with open(filename, 'w') as file:
                file.write(content_d)
        filename = dirname + self._getFilename(self.META) + ".txt"
        with open(filename, 'w') as file:
            content_meta = json.dumps(content[self.META])
            file.write(content_meta)

    def _getFilename(self, name):
        filename = name.replace(' ', '-').replace('/', '-')
        return "".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ']).rstrip()
