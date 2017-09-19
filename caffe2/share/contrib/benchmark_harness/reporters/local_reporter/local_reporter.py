#!/usr/bin/env python3

from arg_parse import getParser
from reporters.reporter_base import ReporterBase

import json
import os
import shutil

getParser().add_argument("--local_reporter", action="store_true",
    help="Save the result to a file")

class LocalReporter(ReporterBase):
    DETAILS = 'details'
    def __init__(self):
        super(LocalReporter, self).__init__()

    def report(self, content):
        if os.path.isdir(self.DETAILS):
            shutil.rmtree(self.DETAILS, True)
        os.mkdir(self.DETAILS)
        for i in range(len(content)):
            filename = self.DETAILS + "/" + str(i) + ".txt"
            content_i = json.dumps(content[i])
            with open(filename, 'w') as file:
                file.write(content_i)
