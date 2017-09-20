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
        net_name = content[self.SUMMARY][self.NET_NAME]
        netdir = self._getFilename(net_name) + "/"
        platform_name = content[self.SUMMARY][self.PLATFORM]
        platformdir = self._getFilename(platform_name) + "/"
        ts = float(content[self.SUMMARY]['time'])
        dt = datetime.datetime.fromtimestamp(ts)
        datedir = str(dt.year) + "/" + str(dt.month) + "/" + str(dt.day) + "/"
        dirname = platformdir + netdir + datedir
        if getArgs().local_reporter:
            dirname = getArgs().local_reporter + "/" + dirname
        i = 0
        while os.path.exists(dirname + str(i)):
            i = i+1
        dirname = dirname + str(i) + "/"
        os.makedirs(dirname)
        details = content[self.DETAILS]
        for d in details:
            filename = dirname + self._getFilename(d) + ".txt"
            content_d = json.dumps(details[d])
            with open(filename, 'w') as file:
                file.write(content_d)
        filename = dirname + self._getFilename(self.SUMMARY) + ".txt"
        with open(filename, 'w') as file:
            content_summary = json.dumps(content[self.SUMMARY])
            file.write(content_summary)

    def _getFilename(self, name):
        filename = name.replace(' ', '-').replace('/', '-')
        return "".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ']).rstrip()
