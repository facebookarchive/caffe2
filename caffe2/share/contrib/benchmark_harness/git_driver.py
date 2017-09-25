#!/usr/bin/env python

import os
import shutil
import time
from utils.arg_parse import getParser, getArgs, parse
from utils.git import Git
from utils.custom_logger import getLogger
from utils.subprocess_with_logger import processRun

getParser().add_argument("--config", required=True,
    help="The test config file containing all the tests to run")
getParser().add_argument("--tests_dir", required=True,
    help="The root directory that all tests resides.")
getParser().add_argument("--git_dir", required=True,
    help="The base git directory.")
getParser().add_argument("--git_commit",
    help="The git commit on this benchmark run.")
getParser().add_argument("--host", action="store_true",
    help="Run the benchmark on the host.")
getParser().add_argument("--android", action="store_true",
    help="Run the benchmark on all collected android devices.")
getParser().add_argument("--local_reporter",
    help="Save the result to a directory specified by this argument.")
getParser().add_argument("--interval", type=int,
    help="The minimum time interval in seconds between two benchmark runs.")
getParser().add_argument("--status_file",
    help="A file to inform the driver stops running when the content of the file is 0.")
getParser().add_argument("--remote_reporter",
    help="Save the result to a remote server. "
    "The style is domain_name>/<endpoint>|<category>")
getParser().add_argument("--remote_access_token",
    help="The access token to access the remote server")

class GitDriver(object):
    def __init__(self):
        parse()
        self.git = Git(getArgs().git_dir)
        self.commit_hash = None

    def _setupGit(self):
        if getArgs().git_commit:
            self.git.pull("sf", "benchmarking")
            self.git.checkout(getArgs().git_commit)
            new_commit_hash = self.git.run('rev-parse', 'HEAD')
            if new_commit_hash == self.commit_hash:
                return False
            self.commit_hash = new_commit_hash
            if getArgs().android:
                # shutil.rmtree(getArgs().git_dir + "/build_android")
                build_android = getArgs().git_dir + "/scripts/build_android.sh"
                processRun(build_android)
            if getArgs().host:
                # shutil.rmtree(getArgs().git_dir + "/build")
                build_local = getArgs().git_dir + "/scripts/build_local.sh"
                processRun(build_local)
            return True

    def _processConfig(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(getArgs().config, 'r') as file:
            content = file.read().splitlines()
            configs = [x.strip().replace('<test_dir>', getArgs().tests_dir) for x in content]
            configs = [(dir_path + "/harness.py " + x +
                " --exec_base_dir " + getArgs().git_dir +
                (" --android" if getArgs().android else "") +
                (" --host" if getArgs().host else "") +
                (" --local_reporter " + getArgs().local_reporter if getArgs().local_reporter else "") +
                (" --remote_reporter \"" + getArgs().remote_reporter  + "\"" if getArgs().remote_reporter else "") +
                (" --remote_access_token \"" + getArgs().remote_access_token  + "\"" if getArgs().remote_access_token else "") +
                (" --git_commit " + self.commit_hash if self.commit_hash else "")).strip()
                for x in configs]
        return configs

    def runOnce(self):
        if not self._setupGit():
            getLogger().info("No new commit, sleeping...")
            return
        configs = self._processConfig()
        for config in configs:
            cmds = config.split(' ')
            cmd = [x.strip() for x in cmds]
            getLogger().info("Running: %s", ' '.join(cmd))
            # cannot use subprocess because it conflicts with requests
            os.system(' '.join(cmd))

    def run(self):
        if not getArgs().interval:
            getLogger().info("Single run...")
            self.runOnce()
            return
        getLogger().info("Continuous run...")
        interval = getArgs().interval
        while True:
            if getArgs().status_file:
                with open(getArgs().status_file, 'r') as file:
                    content = file.read().strip()
                    if content == "0":
                        getLogger().info("Existing...")
                        return
            prev_ts = time.time()
            self.runOnce()
            current_ts = time.time()
            if current_ts < prev_ts + interval:
                time.sleep(prev_ts + interval - current_ts)

if __name__ == "__main__":
    app = GitDriver()
    app.run()
