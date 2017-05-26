from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import scope, core
from caffe2.proto import caffe2_pb2

import unittest
import threading
import time

SUCCESS_COUNT = 0


def thread_runner(idx, testobj):
    global SUCCESS_COUNT
    testobj.assertEqual(scope.CurrentNameScope(), "")
    testobj.assertEqual(scope.CurrentDeviceScope(), None)
    namescope = "namescope_{}".format(idx)
    dsc = core.DeviceOption(caffe2_pb2.CUDA, idx)
    with scope.DeviceScope(dsc):
        with scope.NameScope(namescope):
            testobj.assertEqual(scope.CurrentNameScope(), namescope + "/")
            testobj.assertEqual(scope.CurrentDeviceScope(), dsc)

            time.sleep(0.01 + idx * 0.01)
            testobj.assertEqual(scope.CurrentNameScope(), namescope + "/")
            testobj.assertEqual(scope.CurrentDeviceScope(), dsc)

    testobj.assertEqual(scope.CurrentNameScope(), "")
    testobj.assertEqual(scope.CurrentDeviceScope(), None)
    SUCCESS_COUNT += 1


class TestScope(unittest.TestCase):

    def testNamescopeBasic(self):
        self.assertEqual(scope.CurrentNameScope(), "")

        with scope.NameScope("test_scope"):
            self.assertEqual(scope.CurrentNameScope(), "test_scope/")

        self.assertEqual(scope.CurrentNameScope(), "")

    def testNamescopeAssertion(self):
        self.assertEqual(scope.CurrentNameScope(), "")

        try:
            with scope.NameScope("test_scope"):
                self.assertEqual(scope.CurrentNameScope(), "test_scope/")
                raise Exception()
        except Exception:
            pass

        self.assertEqual(scope.CurrentNameScope(), "")

    def testDevicescopeBasic(self):
        self.assertEqual(scope.CurrentDeviceScope(), None)

        dsc = core.DeviceOption(caffe2_pb2.CUDA, 9)
        with scope.DeviceScope(dsc):
            self.assertEqual(scope.CurrentDeviceScope(), dsc)

        self.assertEqual(scope.CurrentDeviceScope(), None)

    def testDevicescopeAssertion(self):
        self.assertEqual(scope.CurrentDeviceScope(), None)

        dsc = core.DeviceOption(caffe2_pb2.CUDA, 9)

        try:
            with scope.DeviceScope(dsc):
                self.assertEqual(scope.CurrentDeviceScope(), dsc)
                raise Exception()
        except Exception:
            pass

        self.assertEqual(scope.CurrentDeviceScope(), None)

    def testMultiThreaded(self):
        """
        Test that name/device scope are properly local to the thread
        and don't interfere
        """
        global SUCCESS_COUNT
        self.assertEqual(scope.CurrentNameScope(), "")
        self.assertEqual(scope.CurrentDeviceScope(), None)

        threads = []
        for i in range(4):
            threads.append(threading.Thread(
                target=thread_runner,
                args=(i, self),
            ))
        for t in threads:
            t.start()

        with scope.NameScope("master"):
            self.assertEqual(scope.CurrentDeviceScope(), None)
            self.assertEqual(scope.CurrentNameScope(), "master/")
            for t in threads:
                t.join()

            self.assertEqual(scope.CurrentNameScope(), "master/")
            self.assertEqual(scope.CurrentDeviceScope(), None)

        # Ensure all threads succeeded
        self.assertEqual(SUCCESS_COUNT, 4)
