#!/usr/bin/env python3

from platforms.platform_base import PlatformBase
import os.path as path

class AndroidPlatform(PlatformBase):
    def __init__(self, adb, args):
        super(AndroidPlatform, self).__init__(args)
        self.adb = adb

    def setupPlatform(self):
        self.adb.push(self.args.net)
        self.adb.push(self.args.init_net)
        if self.args.input_file:
            self.adb.push(self.args.input_file)
        self.adb.push(self.args.program)

    def runBenchmark(self):
        basename = path.basename(self.args.program)
        program = self.adb.dir + basename
        init_net = path.basename(self.args.init_net)
        net = path.basename(self.args.net)
        cmd = ["cd", self.adb.dir, "&&", program,
            "--init_net", init_net,
            "--net", net,
            "--input", self.args.input,
            "--warmup", str(self.args.warmup),
            "--iter", str(self.args.iter),
            ]
        if self.args.input_file:
            input_file = path.basename(self.args.input_file)
            cmd.extend(["--input_file", input_file])
        if self.args.input_dims:
            cmd.extend(["--input_dims", self.args.input_dims])
        if self.args.output:
            cmd.extend(["--output", self.args.output])
            self.adb.shell(["rm", "-rf", self.adb.dir + "output"])
            cmd.extend(["--output_folder", self.adb.dir + "output"])
        if self.args.run_individual:
            cmd.extend(["--run_individual", "true"])
        self.adb.shell(cmd)

    def collectData(self):
        pass
