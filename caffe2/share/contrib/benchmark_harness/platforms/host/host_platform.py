#!/usr/bin/env python3

from platforms.platform_base import PlatformBase
import logging
import subprocess

logger = logging.getLogger(__name__)

class HostPlatform(PlatformBase):
    def __init__(self, args):
        super(HostPlatform, self).__init__(args)

    def setupPlatform(self):
        pass

    def runBenchmark(self):
        cmd = [
            self.args.program,
            "--logtostderr", "1",
            "--init_net", self.args.init_net,
            "--net", self.args.net,
            "--input", self.args.input,
            "--warmup", self.args.warmup,
            "--iter", self.args.iter,
            ]
        if self.args.input_file:
            cmd.extend(["--input_file", self.args.input_file])
        if self.args.input_dims:
            cmd.extend(["--input_dims", self.args.input_dims])
        if self.args.output:
            cmd.extend(["--output", self.args.output])
            cmd.extend(["--output_folder", self.args.output_folder + "output"])
        if self.args.run_individual:
            cmd.extend(["--run_individual", "true"])
        command = ' '.join(cmd)
        logger.log(logging.INFO,
                    "Running: %s",
                    command)
        return subprocess.check_output(cmd)

    def collectData(self):
        pass
