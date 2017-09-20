#!/usr/bin/env python3

from arg_parse import getParser, getArgs
from reporters.reporter_base import ReporterBase

import json
import os
import shutil

getParser().add_argument("--local_reporter", action="store_true",
    help="Save the result to a file.")
getParser().add_argument("--output_dir",
    help="The directory to save the benchmark output.")


class LocalReporter(ReporterBase):
    def __init__(self):
        super(LocalReporter, self).__init__()

    def report(self, content):
        net_name = content[self.SUMMARY][self.NET_NAME]
        netdir = self.getFilename(net_name)
        platform_name = content[self.SUMMARY][self.PLATFORM]
        platformdir = self.getFilename(platform_name)
        dirname = platformdir + "/" + netdir + "/"
        if getArgs().output_dir:
            dirname = getArgs().output_dir + "/" + dirname
        if os.path.isdir(dirname):
            shutil.rmtree(dirname, True)
        os.makedirs(dirname)
        details = content[self.DETAILS]
        for d in details:
            filename = dirname + self.getFilename(d) + ".txt"
            content_d = json.dumps(details[d])
            with open(filename, 'w') as file:
                file.write(content_d)
        filename = dirname + self.getFilename(self.SUMMARY) + ".txt"
        with open(filename, 'w') as file:
            content_summary = json.dumps(content[self.SUMMARY])
            file.write(content_summary)

    def getFilename(self, name):
        filename = name.replace(' ', '-').replace('/', '-')
        return "".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ']).rstrip()
