#!/usr/bin/env python3

from adb import ADB
from android_platform import AndroidPlatform

class AndroidDriver:
    def __init__(self, args, devices = None):
        self.args = args
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
            platforms.append(AndroidPlatform(adb, self.args))
        return platforms
