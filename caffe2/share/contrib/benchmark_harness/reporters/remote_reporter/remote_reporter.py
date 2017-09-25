#!/usr/bin/env python

from utils.arg_parse import getParser, getArgs
from reporters.reporter_base import ReporterBase
from utils.custom_logger import getLogger

import json
import requests
import urllib

getParser().add_argument("--remote_reporter",
    help="Save the result to a remote server. "
    "The style is <domain_name>/<endpoint>|<category>")
getParser().add_argument("--remote_access_token",
    help="The access token to access the remote server")

class RemoteReporter(ReporterBase):
    def __init__(self):
        super(RemoteReporter, self).__init__()

    def report(self, content):
        if not getArgs().remote_reporter:
            return
        access_token = getArgs().remote_access_token
        remote = self._getRemoteInfo()
        logs = self._composeMessages(content, remote['category'])

        self._log(remote['url'], access_token, logs)

    def _merge_dicts(self, *dict_args):
        """
        Given any number of dicts, shallow copy and merge into a new dict,
        precedence goes to key value pairs in latter dicts.
        """
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def _getRemoteInfo(self):
        endpoint = getArgs().remote_reporter.strip().split("|")
        assert len(endpoint) == 2, "Category not speied in remote endpoint"
        res = {}
        res['url'] = endpoint[0].strip()
        if len(res['url']) < 5 or res['url'][:4] != 'http':
            res['url'] = 'https://' + res['url']
        res['category'] = endpoint[1].strip()

        return res

    def _composeMessages(self, content, category):
        logs = []
        meta = content[self.META]
        ts = int(meta['time'])
        meta.pop('time', None)
        for item in content[self.DATA]:
            data = content[self.DATA][item]
            new_meta = meta.copy()
            new_meta['type'] = item
            summary = {'time': ts}
            for k in data['summary']:
                summary[k] = int(data['summary'][k] * 1000)
            values = data['values']
            message = {
                'int' : summary,
                'normal' : new_meta,
                'normvector' : {'values' : values},
            }
            message_string = json.dumps(message, sort_keys=True)
            log = {
                'category': category,
                'message': message_string,
                'line_escape': False,  # This allows double quotes, back slashes, etc.
                                       # to work correctly.
            }
            logs.append(log)
        return logs

    def _log(self, url, access_token, logs):
        num_logs = len(logs)
        logs_string = json.dumps(logs, sort_keys=True)
        parameters = {
            'access_token': access_token,
            'logs': logs_string,
        }
        request = requests.post(url, json=parameters)
        result = request.json()
        count_key = 'count'
        is_good = request.ok and count_key in result and result[count_key] == num_logs
        if not is_good:
            getLogger().error("Submit data to remote server failed")
            if not request.ok:
                getLogger().error("Request is not okay")
            elif not count_key in result:
                getLogger().error("%s is not in request return value", count_key)
            else:
                getLogger().error("Sent %d records out of a total of %d", result[count_key], num_logs)
        else:
            getLogger().info("Sent %d records to remote server %s successfully.", num_logs, url)
        return is_good
