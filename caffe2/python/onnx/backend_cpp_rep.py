# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

## @package onnx
# Module caffe2.python.onnx.backend_rep_cpp

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx.backend.base import BackendRep, namedtupledict
import caffe2.python._import_c_extension as C

# This is a wrapper around C++ Caffe2BackendRep,
# mainly to handle the different input and output types for convenience of Python
class Caffe2CppRep(BackendRep):
    def __init__(self, cpp_rep):
        super(Caffe2CppRep, self).__init__()
        self.__core = cpp_rep
        self.__external_outputs = cpp_rep.external_outputs()
        self.__external_inputs = cpp_rep.external_inputs()
        self.__uninitialized_inputs = cpp_rep.uninitialized_inputs()

    def run(self, inputs):
        output_values = None
        if isinstance(inputs, dict):
            output_values = self.__core.run(inputs)
        elif isinstance(inputs, list) or isinstance(inputs, tuple):
            if len(inputs) != len(self.__uninitialized_inputs):
                raise RuntimeError('Expected {} values for uninitialized '
                                   'graph inputs ({}), but got {}.'.format(
                                        len(self.__uninitialized_inputs),
                                        ', '.join(self.__uninitialized_inputs),
                                        len(inputs)))
            input_map = {}
            for k, v in zip(self.__uninitialized_inputs, inputs):
                input_map[k] = v
            output_values = self.__core.run(input_map)
        else:
            # single input
            output_values = self.__core.run([inputs])
        return namedtupledict('Outputs', self.__external_outputs)(*output_values)

