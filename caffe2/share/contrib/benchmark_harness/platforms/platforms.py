#!/usr/bin/env python3

from android.android_driver import AndroidDriver
from host.host_platform import HostPlatform


def getPlatforms(args):
    platforms = []
    if args.host:
        platforms.append(HostPlatform(args))
    if args.android:
        driver = AndroidDriver(args)
        platforms.extend(driver.getAndroidPlatforms())
    if not platforms:
        logger.log(logger.ERROR, "No platform is specified.")
    return platforms
