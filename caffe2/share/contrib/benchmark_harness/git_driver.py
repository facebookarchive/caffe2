#!/usr/bin/env python

import datetime
import json
import os
import shutil
import tempfile
import time
from utils.arg_parse import getParser, getArgs, getUnknowns, parseKnown
from utils.git import Git
from utils.custom_logger import getLogger
from utils.subprocess_with_logger import processRun

getParser().add_argument("--config", required=True,
    help="Required. The test config file containing all the tests to run")
getParser().add_argument("--models_dir", required=True,
    help="Required. The root directory that all models resides.")
getParser().add_argument("--git_dir", required=True,
    help="Required. The base git directory.")
getParser().add_argument("--git_commit", default="master",
    help="The git commit this benchmark runs on. It can be a branch. Defaults to master")
getParser().add_argument("--git_base_commit",
    help="In A/B testing, this is the control commit that is used to compare against. " +
    "If not specified, the default is the first commit in the week in UTC timezone. " +
    "Even if specified, the control is the later of the specified commit and the commit at the start of the week.")
getParser().add_argument("--host", action="store_true",
    help="Run the benchmark on the host.")
getParser().add_argument("--android", action="store_true",
    help="Run the benchmark on all connected android devices.")
getParser().add_argument("--interval", type=int,
    help="The minimum time interval in seconds between two benchmark runs.")
getParser().add_argument("--status_file",
    help="A file to inform the driver stops running when the content of the file is 0.")
getParser().add_argument("--git_repository", default="origin",
    help="The remote git repository. Defaults to origin")
getParser().add_argument("--git_branch", default="master",
    help="The remote git repository branch. Defaults to master")

class GitDriver(object):
    def __init__(self):
        parseKnown()
        self.git = Git(getArgs().git_dir)
        self.prev_commit_hash = None
        self.git_info = {}

    def runOnce(self):
        if not self._pullNewCommits():
            return
        tempdir = tempfile.mkdtemp()
        git_info = self._setupGit(tempdir)
        configs = self._processConfig(git_info)
        for config in configs:
            cmds = config.split(' ')
            cmd = [x.strip() for x in cmds]
            getLogger().info("Running: %s", ' '.join(cmd))
            # cannot use subprocess because it conflicts with requests
            os.system(' '.join(cmd))
        shutil.rmtree(tempdir, True)
        getLogger().info("Done oone benchmark run.")

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

    def _pullNewCommits(self):
        self.git.pull(getArgs().git_repository, getArgs().git_branch)
        self.git.checkout(getArgs().git_commit)
        new_commit_hash = self.git.getCommitHash(getArgs().git_commit)
        if new_commit_hash == self.prev_commit_hash:
            getLogger().info("Commit %s is already processed, sleeping...", new_commit_hash)
            return False
        self.prev_commit_hash = new_commit_hash
        return True

    def _setupGit(self, tempdir):
        git_info = {}
        treatment_dir = tempdir + '/treatment'
        os.mkdir(treatment_dir)
        git_info_treatment = self._setupGitStep(treatment_dir, getArgs().git_commit)
        git_info['treatment'] = git_info_treatment
        # figure out the base commit. It is the first commit in the week
        control_commit_hash = self._getControlCommit(git_info_treatment['commit_time'], getArgs().git_base_commit)

        control_dir = tempdir + '/control'
        os.mkdir(control_dir)
        git_info['control'] = self._setupGitStep(control_dir, control_commit_hash)
        return git_info

    def _setupGitStep(self, tempdir, commit):
        git_info = {}
        self.git.checkout(commit)
        git_info['commit'] = self.git.getCommitHash(commit)
        git_info['commit_time'] = self.git.getCommitTime(git_info['commit'])
        self._buildProgram(tempdir, git_info)
        return git_info

    def _buildProgram(self, tempdir, git_info):
        if getArgs().android:
            # shutil.rmtree(getArgs().git_dir + "/build_android")
            build_android = getArgs().git_dir + "/scripts/build_android.sh"
            processRun(build_android)
            src = getArgs().git_dir + \
                '/build_android/caffe2/share/contrib/binaries/caffe2_benchmark/binaries/caffe2_benchmark'
            dst = tempdir + '/caffe2_benchmark_android'
            shutil.copyfile(src, dst)
            git_info['program_android'] = dst
        if getArgs().host:
            # shutil.rmtree(getArgs().git_dir + "/build")
            build_local = getArgs().git_dir + "/scripts/build_local.sh"
            src = getArgs().git_dir + \
                '/build/caffe2/share/contrib/binaries/caffe2_benchmark/binaries/caffe2_benchmark'
            dst = tempdir + '/caffe2_benchmark_host'
            shutil.copyfile(src, dst)
            git_info['program_host'] = dst
            processRun(build_local)

    def _processConfig(self, git_info):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        unknowns = getUnknowns()
        with open(getArgs().config, 'r') as file:
            content = file.read().splitlines()
            configs = [x.strip().replace('<models_dir>', getArgs().models_dir) for x in content]
            configs = [(dir_path + "/harness.py " + x +
                (" --android" if getArgs().android else "") +
                (" --host" if getArgs().host else "") +
                (" --git_info \'" + json.dumps(git_ifno) + "\'")
                (" --git_commit " + self.commit_hash) +
                (" --git_commit_time " + self.commit_hash_time)).strip() + " " +
                ' '.join(['"' + x + '"' for x in unknowns])
                for x in configs]
        return configs

    def _getControlCommit(self, reference_time, base_commit):
        # Get start of week
        dt = datetime.datetime.utcfromtimestamp(reference_time)
        monday = dt - datetime.timedelta(days=dt.weekday())
        start_of_week = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        ut_start_of_week = time.mktime(start_of_week.utctimetuple())

        if base_commit:
            base_commit_time = self.git.getCommitTime(base_commit)
            if base_commit_time >= ut_start_of_week:
                return base_commit

        # Give more buffer to the date range to avoid the timezone issues
        start = start_of_week - datetime.timedelta(days=1)
        end = dt + datetime.timedelta(days=1)
        logs_str = self.git.run('log', '--after', start.isoformat(), '--before', end.isoformat(), '--reverse', '--pretty=format:%H:%ct')
        logs = logs_str.split('\n')
        for row in logs:
            items = row.strip().split(':')
            assert len(items) == 2, "Git log format is wrong"
            commit_hash = items[0].strip()
            unix_time = int(items[1].strip())
            if unix_time >= ut_start_of_week:
                return commit_hash
        assert False, "Cannot find the control commit"
        return None

if __name__ == "__main__":
    app = GitDriver()
    app.run()
