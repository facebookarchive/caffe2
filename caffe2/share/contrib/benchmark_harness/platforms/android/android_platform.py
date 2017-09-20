#!/usr/bin/env python

from platforms.platform_base import PlatformBase
import os.path as path

from utils.arg_parse import getArgs

class AndroidPlatform(PlatformBase):
    def __init__(self, adb):
        super(AndroidPlatform, self).__init__()
        self.adb = adb
        self.output = None

    def setupPlatform(self):
        try:
            self.adb.logcat("-G", "1M")
        except Exception:
            self.adb.logcat("-G", "256K")
        self.adb.logcat('-b', 'all', '-c')
        self.adb.push(getArgs().net)
        self.adb.push(getArgs().init_net)
        if getArgs().input_file:
            self.adb.push(getArgs().input_file)

        self.adb.push(self._getProgram())

    def runBenchmark(self):
        basename = path.basename(self._getProgram())
        program = self.adb.dir + basename
        init_net = path.basename(getArgs().init_net)
        net = path.basename(getArgs().net)
        cmd = ["cd", self.adb.dir, "&&", program,
            "--init_net", init_net,
            "--net", net,
            "--input", getArgs().input,
            "--warmup", str(getArgs().warmup),
            "--iter", str(getArgs().iter),
            ]
        if getArgs().input_file:
            input_file = path.basename(getArgs().input_file)
            cmd.extend(["--input_file", input_file])
        if getArgs().input_dims:
            cmd.extend(["--input_dims", getArgs().input_dims])
        if getArgs().output:
            cmd.extend(["--output", getArgs().output])
            self.adb.shell(["rm", "-rf", self.adb.dir + "output"])
            cmd.extend(["--output_folder", self.adb.dir + "output"])
        if getArgs().run_individual:
            cmd.extend(["--run_individual", "true"])

        self.adb.shell(cmd)
        self.output = self.adb.logcat('-d')

    def collectData(self):
        result = super(AndroidPlatform, self).collectData()
        arch = self.adb.shell(['getprop', 'ro.product.model'])
        result[self.SUMMARY][self.PLATFORM] = arch
        return result

    def _getProgram(self):
        if getArgs().program:
            program = getArgs().program
        elif getArgs().exec_base_dir:
            program = getArgs().exec_base_dir + \
                '/build_android/caffe2/share/contrib/binaries/benchmark/binaries/benchmark'
        return program
