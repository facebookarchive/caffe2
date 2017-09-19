#!/usr/bin/env python3

from adb import ADB
from android_platform import AndroidPlatform
from arg_parse import getArgs, getParser

getParser().add_argument("--android", action="store_true",
    help="Run the benchmark on all collected android devices.")

class AndroidDriver:
    def __init__(self, devices = None):
        self.adb = ADB()
        if devices:
            if isinstance(devices, string):
                devices = [devices]
        self.devices = devices

    def getDevices(self):
        devices_str = self.adb.run("devices", "-l")
        rows = devices_str.split('\n')
        rows.pop(0)
        devices = []
        for row in rows:
            if row.strip():
                items = row.split(' ')
                device_id = items[0].strip()
                devices.append(device_id)
        return devices


    def getAndroidPlatforms(self):
        if self.devices is None:
            self.devices = self.getDevices()
        platforms = []
        for device in self.devices:
            adb = ADB(device)
            platforms.append(AndroidPlatform(adb))
        return platforms
