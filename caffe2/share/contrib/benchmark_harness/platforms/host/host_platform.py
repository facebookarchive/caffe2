#!/usr/bin/env python

from platforms.platform_base import PlatformBase
import subprocess
import platform
from utils.arg_parse import getArgs, getParser
from utils.custom_logger import getLogger

getParser().add_argument("--host", action="store_true",
    help="Run the benchmark on the host.")

class HostPlatform(PlatformBase):
    def __init__(self):
        super(HostPlatform, self).__init__()
        self.output = None

    def setupPlatform(self):
        pass

    def runBenchmark(self):
        cmd = [
            getArgs().program,
            "--logtostderr", "1",
            "--init_net", getArgs().init_net,
            "--net", getArgs().net,
            "--input", getArgs().input,
            "--warmup", str(getArgs().warmup),
            "--iter", str(getArgs().iter),
            ]
        if getArgs().input_file:
            cmd.extend(["--input_file", getArgs().input_file])
        if getArgs().input_dims:
            cmd.extend(["--input_dims", getArgs().input_dims])
        if getArgs().output:
            cmd.extend(["--output", getArgs().output])
            cmd.extend(["--output_folder", getArgs().output_folder + "output"])
        if getArgs().run_individual:
            cmd.extend(["--run_individual", "true"])
        command = ' '.join(cmd)
        getLogger().info("Running: %s", command)
        pipes = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        std_out, std_err = pipes.communicate()
        assert pipes.returncode == 0, "Benchmark run failed"
        if len(std_err):
            self.output = std_err

    def collectData(self):
        result = super(HostPlatform, self).collectData()
        arch = platform.processor()
        result[self.SUMMARY][self.PLATFORM] = arch
        return result
