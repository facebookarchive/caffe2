#!/usr/bin/env python3

from android.android_driver import AndroidDriver
from host.host_platform import HostPlatform
from arg_parse import getArgs
from custom_logger import getLogger

def getPlatforms():
    platforms = []
    if getArgs().host:
        platforms.append(HostPlatform())
    if getArgs().android:
        driver = AndroidDriver()
        platforms.extend(driver.getAndroidPlatforms())
    if not platforms:
        getLogger().error("No platform is specified.")
    return platforms
